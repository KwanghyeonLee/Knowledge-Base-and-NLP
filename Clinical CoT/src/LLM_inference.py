import argparse
import asyncio
import json
import yaml
import os
import random
from copy import deepcopy
import gzip
import pickle

import numpy as np
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAIChat, OpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

TOTAL_COST = 0  # making this a global variable, be aware this may lead to issues in concurrent scenarios
os.environ["OPENAI_ORGANIZATION"] = "Your/organization/id"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--model_name", choices=['gpt-3.5-turbo', 'gpt-4'], default='gpt-4')
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--prompt_key", type=str, default=None)
    parser.add_argument("--save_dir", type=str, required=True,
                        help="It should be a NEW DIRECTORY. Please do not use an existing")
    parser.add_argument("--num_sample", type=int, default=None,
                        help="If you want to test your code by sampling a small number of data, you can set this argument.")
    # parser.add_argument("--min_context_len", type=int, default=0)
    args = parser.parse_args()
    if args.num_sample:
        args.save_dir = args.save_dir + f"_sample{args.num_sample}"
    return args


def load_prompt(args):
    with open(args.prompt, "r", encoding="UTF-8") as f:
        prompt = yaml.load(f, Loader=yaml.FullLoader)[args.prompt_key]
    return prompt


def prepare_model_input(prompt, data_path):
    # text data (json)
    with open(data_path, "r", encoding="UTF-8") as f:
        data = json.load(f)

    all_model_data = []
    for d in tqdm(data):
        input_temp = dict()
        # I know this code is inefficient, only for clarity
        input_temp['id'], input_temp['patient_data'], input_temp['label'] = d['id'], d['patient_data'], d['label']
        input_temp['model_input'] = prompt.format(**d)
        all_model_data.append(input_temp)

    return all_model_data


def load_and_prepare_data(args):
    prompt = load_prompt(args)
    print("Preparing model inputs...")
    all_model_data = prepare_model_input(
        prompt, args.input_path)
    return all_model_data


def sample_indices(all_model_data, num_sample):
    random.seed(0)
    cand_indices = list(range(len(all_model_data)))
    sampled_indices = random.sample(cand_indices, num_sample)
    return sampled_indices


def filter_data(all_model_data, num_sample):
    if num_sample:
        sampled_indices = sample_indices(all_model_data, num_sample)
        all_model_data = [all_model_data[i] for i in sampled_indices]
    return all_model_data


async def async_generate(llm, model_data, idx, save_dir, args):
    global TOTAL_COST
    system_message = SystemMessage(content=model_data['model_input'])
    # human_message = HumanMessage(content=model_input) # if you need it
    while True:
        try:
            response = await llm.agenerate([[system_message]])
            # response = await llm.agenerate([[system_message, human_message]]) # if you need it
            token_used = response.llm_output['token_usage']['total_tokens']
            
            if args.model_name == 'gpt-3.5-turbo':
                TOTAL_COST += token_used / 1000 * 0.002
            else:
                TOTAL_COST += token_used / 1000 * 0.06
            print(idx, TOTAL_COST)
            break
        except Exception as e:
            print(f"Exception occurred: {e}")
            response = None

    result = deepcopy(model_data)
    result['prediction'] = response.generations[0][0].text
    with open(os.path.join(save_dir, f"{idx}.json"), "w", encoding='UTF-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    return result


async def generate_concurrently(all_model_data, start_idx, save_dir, args):
    llm = ChatOpenAI(model_name=args.model_name,  # 'gpt-3.5-turbo' or 'gpt-4'
                     openai_api_key="your/openai/api/key",
                     temperature=0.7, max_tokens=2000, max_retries=100)
    tasks = [async_generate(llm, model_data, i+start_idx, save_dir, args)
             for i, model_data in enumerate(all_model_data)]
    return await tqdm_asyncio.gather(*tasks)


async def main(args):
    all_model_data = load_and_prepare_data(args)
    all_model_data = filter_data(all_model_data, args.num_sample)

    # Check if the save_dir exists
    if os.path.exists(args.save_dir):
        print("The save_dir already exists. Please change the save_dir.")

    os.makedirs(args.save_dir, exist_ok=True)
    all_results = []
    if len(all_model_data) > 300:
        for start_idx in tqdm(range(0, len(all_model_data), 300)):
            cur_model_data = all_model_data[start_idx:start_idx+300]
            all_results.extend(await generate_concurrently(cur_model_data, start_idx, args.save_dir, args))
    else:
        all_results = await generate_concurrently(all_model_data, 0, args.save_dir, args)

    total_result_path = args.save_dir + "_total_results.json"
    with open(os.path.join(total_result_path), "w", encoding='UTF-8') as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
