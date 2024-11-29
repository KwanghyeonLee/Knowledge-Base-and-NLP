# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional
import argparse
import os
import dill
import sys

from tqdm.asyncio import tqdm_asyncio
from langchain.llms import OpenAIChat, OpenAI
from torch.nn.utils.rnn import pad_sequence
import asyncio
import torch
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig, AutoPeftModelForCausalLM
from tqdm import tqdm
from transformers import Adafactor, AutoTokenizer, HfArgumentParser, pipeline, BitsAndBytesConfig
import json
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler
import evaluate
if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback
    from trl import AutoModelForCausalLMWithValueHead
from datasets import load_dataset, concatenate_datasets
import openai
from evaluate import load
import subprocess
import os
from models import OpenAIModel, VllmModel
from utils.utils import extract_python_code
from utils.path import save_json_file
from utils.eval import *

os.environ["HF_ALLOW_CODE_EVAL"] = "1"
tqdm.pandas()

TEST_CASE_PATH = "PATH"


@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """

    # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
    # models like gpt-neo* models are more suitable.
    model_name: Optional[str] = field(default="", metadata={"help": "the model name"})
    tokenizer_name: Optional[str] = field(default="", metadata={"help": "the tokenizer name"})
    reward_model_name: Optional[str] = field(default="", metadata={"help": "the reward model name"})
    log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
    output_max_length: Optional[int] = field(default=128, metadata={"help": "maximum length for generation"})
    mini_batch_size: Optional[int] = field(default=1, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=32, metadata={"help": "the batch size"})
    ppo_epochs: Optional[int] = field(default=4, metadata={"help": "the number of ppo epochs"})
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "the number of gradient accumulation steps"}
    )
    adafactor: Optional[bool] = field(default=False, metadata={"help": "whether to use the adafactor optimizer"})
    early_stopping: Optional[bool] = field(default=False, metadata={"help": "whether to early stop"})
    target_kl: Optional[float] = field(default=0.1, metadata={"help": "kl target for early stopping"})
    reward_baseline: Optional[float] = field(
        default=0.0,
        metadata={"help": "a baseline value that is subtracted from the reward"},
    )
    batched_gen: Optional[bool] = field(default=False, metadata={"help": "whether to use the batched text gen"})
    save_freq: Optional[int] = field(default=None, metadata={"help": "n steps to save the model"})
    output_dir: Optional[str] = field(default="runs/", metadata={"help": "n steps to save the model"})
    seed: Optional[int] = field(default=0, metadata={"help": "the seed"})
    steps: Optional[int] = field(default=20000, metadata={"help": "number of epochs"})
    init_kl_coef: Optional[float] = field(
        default=0.2,
        metadata={"help": "Initial KL penalty coefficient (used for adaptive and linear control)"},
    )

    adap_kl_ctrl: Optional[bool] = field(default=True, metadata={"help": "Use adaptive KL control, otherwise linear"})
    reward_model_name: Optional[str] = field(default="PATH")
    reward_model_port: Optional[int] = field(default=8000)

    reward_model_url: Optional[str] = field(default="url")

async def main():
    parser = HfArgumentParser(ScriptArguments)
    script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]
    reward_model_name = script_args.reward_model_name
    #dataset_name = "lvwerra/stack-exchange-paired"
    config = PPOConfig(
        steps=script_args.steps,
        model_name=script_args.model_name,
        learning_rate=script_args.learning_rate,
        log_with=script_args.log_with,
        batch_size=script_args.batch_size,
        mini_batch_size=script_args.mini_batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        optimize_cuda_cache=True,
        early_stopping=script_args.early_stopping,
        target_kl=script_args.target_kl,
        ppo_epochs=script_args.ppo_epochs,
        seed=script_args.seed,
        init_kl_coef=script_args.init_kl_coef,
        adap_kl_ctrl=script_args.adap_kl_ctrl,
        remove_unused_columns=False
    )



    # We then define the arguments to pass to the sentiment analysis pipeline.
    # We set `return_all_scores` to True to get the sentiment score for each token.
    sent_kwargs = {
        "return_all_scores": True,
        "function_to_apply": "none",
        "batch_size": 16,
        "truncation": True,
    }

    print("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    # GPT-2 tokenizer has a pad token, but it is not eos_token by default. We need to set it to eos_token.
    # only for this model.

    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token




    # def collator(data):
    #     return dict((key, [d[key] for d in data]) for key in data[0])

    dataset = load_dataset("DATA")
    raw_trainset = dataset['train'] 
    raw_evalset = dataset['eval']

    problem_ids_to_index = {pi:i for i, pi in enumerate(list(set(raw_trainset['problem_id']+raw_evalset['problem_id'])))}
    problem_index_to_id = {v:k for k,v in problem_ids_to_index.items()}
    dataset_keys = list(raw_evalset[0].keys())
    def format_dataset(dataset):
        instruction = "Provide feedback on the errors in the given code and suggest the correct code to address the described problem."
        original_columns = dataset.column_names
        def preprocess_function(examples):
            new_examples = {
                "input_ids": [],
                "instance_index": [],
                "query": [],
            }
            # print(examples)
        #       input_variables: 
        # - "description"
        # - "output_format"
        # - "input_format"
        # - "wrong_code"


            dummy_feedback_model = VllmModel("MODEL_NAME")
            feedback_prompt = dummy_feedback_model.get_prompt_template("feedback")
            for i in range(len(examples['problem_id'])):
                input_string = f"{instruction}\nProblem Description:{examples['description'][i]}\nIncorrect Code:\n{examples['wrong_code'][i]}\n"
                cur_data_dict = {k:examples[k][i] for k in dataset_keys}
                input_string = dummy_feedback_model.apply_function_to_prompt(feedback_prompt, "feedback", **cur_data_dict)
                input_ids = tokenizer(input_string, truncation=True,return_tensors="pt", max_length=2048)['input_ids'].squeeze(0)

                new_examples['input_ids'].append(input_ids)
                new_examples['instance_index'].append(torch.tensor([i]))
                new_examples['query'].append(input_string)
            
            return new_examples

        formatted_dataset = dataset.map(preprocess_function, batched=True, remove_columns=original_columns)
        formatted_dataset.set_format(type="torch")
        return formatted_dataset
    def collator(data):
        return dict((key, [d[key] for d in data]) for key in data[0])
        # return out_dict 

    trainset = format_dataset(raw_trainset)

    evalset = format_dataset(raw_evalset)
    # set seed before initializing value head for deterministic eval
    set_seed(config.seed)

    # Now let's build the model, the reference model, and the tokenizer.
    current_device = Accelerator().local_process_index

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        config.model_name,
        load_in_8bit=True,
        device_map={"": current_device},
        peft_config=lora_config,
    )

    optimizer = None
    if script_args.adafactor:
        optimizer = Adafactor(
            filter(lambda p: p.requires_grad, model.parameters()),
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
            lr=config.learning_rate,
        )
    # We then build the PPOTrainer, passing the model, the reference model, the tokenizer
    ppo_trainer = PPOTrainer(
        config,
        model,
        ref_model=None,
        tokenizer=tokenizer,
        dataset=trainset,
        data_collator=collator,
        optimizer=optimizer,
    )

    # We then build the sentiment analysis pipeline using our reward model, passing the
    # model name and the sentiment analysis pipeline arguments. Let's also make sure to
    # set the device to the same device as the PPOTrainer.
    device = ppo_trainer.accelerator.device
    if ppo_trainer.accelerator.num_processes == 1:
        device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a ` pipeline` bug

    # We then define the arguments to pass to the `generate` function. These arguments
    # are passed to the `generate` function of the PPOTrainer, which is a wrapper around
    # the `generate` function of the trained model.
    generation_kwargs = {
        # "min_length": -1,
        "top_k": 0.0,
        "top_p": 0.95,
        "temperature": 0.4,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "max_new_tokens": script_args.output_max_length,
    }
    output_min_length = 1
    output_max_length = script_args.output_max_length
    output_length_sampler = LengthSampler(output_min_length, output_max_length)
    total_iterations = min(len(ppo_trainer.dataloader), config.total_ppo_epochs)

    ## INIT reward model ##

    reward_model = VllmModel(model_name=script_args.reward_model_name, port=script_args.reward_model_port, url=script_args.reward_model_url)

    with tqdm(enumerate(ppo_trainer.dataloader), total=total_iterations) as pbar:
        for epoch, batch in pbar:
            if epoch >= config.total_ppo_epochs:
                break
            
            # unwrapped_model.gradient_checkpointing_disable()
            # unwrapped_model.config.use_cache = True
            model.eval()
            # prompt = batch["prompt"]
            question_tensors = batch["input_ids"]
            instance_indices = batch['instance_index']
            instance_ids = [int(pi[0]) for pi in instance_indices]
            
            full_data_batch = deepcopy([raw_trainset[ii] for ii in instance_ids])

            # print(problem_ids)
            # import pdb
            # pdb.set_trace()
            # # print(len(question_tensors))
            # print(question_tensors.shape)
            print(question_tensors[0].shape)
            # print(question_tensors)
            response_tensors = ppo_trainer.generate(
                question_tensors,
                return_prompt=False,
                length_sampler=output_length_sampler,
                **generation_kwargs,
            )
            # generated feedback from policy model
            batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
            generated_feedback = batch["response"]
            for fi in range(len(generated_feedback)):
                full_data_batch[fi]['feedbacks'] = [generated_feedback[fi]]

            question_strings = tokenizer.batch_decode(question_tensors, skip_special_tokens=True)
            # Compute reward score (using the sentiment analysis pipeline)
            # texts = [q + r for q, r in zip(batch["query"], batch["response"])]
                
            # outputs = [train.get_constraint(i).check(r.split('Answer: ')[-1], t) for r, t, i in zip(batch['response'], batch['targets'], batch['idx'])]
            # rewards = [torch.tensor([float(o)]).to(question_tensors[0].device) for o in outputs]
            refine_prompt_template = reward_model.get_prompt_template(mode="refine")

            edit_generation_args = {
                    "max_tokens": 2048,
                    "temperature": 0.4,
                    "top_p":0.95,
                    "frequency_penalty": 0.0,
                    "stop_sequence": None
                    }

            edit_generation_args = argparse.Namespace(**edit_generation_args)
            code_results = await process_items_concurrently(reward_model, refine_prompt_template, full_data_batch, edit_generation_args, "refine")

            for eci, edited_code in enumerate(code_results):
                full_data_batch[eci]['refined_codes'] = code_results[eci]['refined_codes']
            # outputs = get_test_score_with_refine_generation(question_strings, batch['response'], problem_ids, script_args)
            outputs = calculate_pass_scores(full_data_batch)

            rewards = [torch.tensor([float(o)]).to(question_tensors[0].device) for o in outputs]

            # unwrapped_model.gradient_checkpointing_enable()
            # unwrapped_model.config.use_cache = False
            model.train()
            for ri in range(len(response_tensors)):
                response_tensors[ri] = response_tensors[ri].to(question_tensors[0].device)
            # print("Questiuon Tensor")
            # print(question_tensors)
            # print("Response Tensor")
            # print(response_tensors)
            # print("Reward Tensor")
            # print(rewards)
            # Run PPO step
            stats = ppo_trainer.step(question_tensors, response_tensors, rewards)
            ppo_trainer.log_stats(stats, batch, rewards)

            if script_args.save_freq and epoch and epoch % script_args.save_freq == 0:
                ppo_trainer.save_pretrained(script_args.output_dir + f"step_{epoch}")

if __name__ == "__main__":
    asyncio.run(main())
