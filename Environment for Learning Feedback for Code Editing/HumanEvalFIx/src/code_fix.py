import argparse
import asyncio
import json
import yaml
import os
import random
from copy import deepcopy
import gzip
import pickle
import copy
import concurrent.futures
import time
import multiprocessing, time
import os
os.environ["HF_ALLOW_CODE_EVAL"] = "1" 
import sys
import re
from evaluate import load
from datasets import load_dataset

import numpy as np
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
import sys
from src.models import OpenAIModel, VllmModel
from langchain_community.llms import OpenAI
# from langchain_community.llms import OpenAIChat
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from pprint import pprint
from eval_tool import evaluate_solutions

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--critic_url", type=str, default="localhost")
    parser.add_argument("--editor_url", type=str, default="localhost")
    parser.add_argument("--critic_port", type=int, default=8001, help="The port number of the critic model")
    parser.add_argument("--editor_port", type=int, default=8000, help="The port number of the editor model")
    parser.add_argument("--critic_model_name", type=str, default="MODEL_NAME")
    parser.add_argument("--editor_model_name", type=str, default="MODEL_NAME")
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--editor_prompt", type=str, default=None, help="the path to yaml file that contains the contents of the prompts")
    parser.add_argument("--critic_prompt", type=str, default=None, help="the path to yaml file that contains the contents of the prompts")
    parser.add_argument("--critic_prompt_key", type=str, default="critic")
    parser.add_argument("--editor_prompt_key", type=str, default="editor")
    parser.add_argument("--save_dir", type=str, required=True, help="It should be a NEW DIRECTORY. Please do not use an existing")
    parser.add_argument("--feedback_gen", type=str, default='False' )
    parser.add_argument("--is_iter", type=str, default='False', help="If you want to iterate feedback /code gen")
    parser.add_argument("--iter_no", type=int, default=3, help="number of iterations you want to take")
    parser.add_argument("--execution_feedback", type=str, default='False', help="use execution feeback?")
    ## generate args ##
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--frequency_penalty", type=float, default=0.0)
    args = parser.parse_args()
    return args

def extract_python_code(text):
    """
    Extracts Python code blocks from a given string that contain text enclosed within
    triple backticks labeled as 'python'.
    
    Parameters:
    - text (str): The input text containing Python code blocks.
    
    Returns:
    - list of str: A list of extracted code blocks.
    """
    try:
        patterns = [
        r"```python(.*?)'''", # Triple single quotes (general)
        r"```python(.*?)```", # Triple single quotes (general)
        r'```python(.*?)```',  # Triple backticks with 'python'
        r'```Python(.*?)```',
        r'```Python(.*?)\'\'\'',
        r'```(.*?)```',        # Triple backticks without a label
        r"'''python(.*?)'''",
        r"'''Python(.*?)'''",# Triple single quotes with 'python'
        r"'''(.*?)'''",        # Triple single quotes without a label
        r'```(.*?)```',        # Triple backticks (general)
        r"'''(.*?)'''",
        ] # Non-greedy match
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)  # Capture multiline code blocks
            if matches:
                return matches[0].strip()
        
        # Strip leading/trailing whitespace from each match and return
        return matches[0].strip()
    except:
        return text.split("```python\n")[-1].split("```")[0]

def load_prompt(args, mode):
    if args.feedback_gen == 'False' and args.execution_feedback == 'False':
        if mode == "editor":
            with open(args.editor_prompt, "r", encoding="UTF-8") as f:
                prompt = yaml.load(f, Loader=yaml.FullLoader)[args.editor_prompt_key + '_vanilla' ]
        else:
            raise ValueError("Invalide prompt key")
    elif args.feedback_gen == 'False' and args.execution_feedback == 'True':
        if mode == "editor":
            with open(args.editor_prompt, "r", encoding="UTF-8") as f:
                prompt = yaml.load(f, Loader=yaml.FullLoader)[args.editor_prompt_key + '_execution_feedback']
        else:
            raise ValueError("Invalide prompt key")
    elif args.feedback_gen == 'True' and args.execution_feedback == 'False':
        if mode == "critic":
            with open(args.critic_prompt, "r", encoding="UTF-8") as f:
                prompt = yaml.load(f, Loader=yaml.FullLoader)[args.critic_prompt_key]
        elif mode == "editor":
            with open(args.editor_prompt, "r", encoding="UTF-8") as f:
                prompt = yaml.load(f, Loader=yaml.FullLoader)[args.editor_prompt_key]
        else:
            raise ValueError("Invalide prompt key") 
    else:
        raise ValueError("Invalide args setting")        

    return prompt


def load_data(args):
    with open(args.input_path, "r", encoding="UTF-8") as f:
        data = json.load(f)
    return data


def format_model_input(prompt, data):
    all_model_data = []
    for d in tqdm(data):
        formatted_prompt = prompt.format(**d)
        all_model_data.append(formatted_prompt)

    return all_model_data


def prepare_feedback_generation_input(args, data):
    prompt = load_prompt(args, mode="critic")
    print("Preparing model inputs...")
    all_model_data = format_model_input(
        prompt, data)
    
    return all_model_data

def prepare_code_editing_input(args, data, generated_feedback):
    # add generated_feedback to data 
    prompt = load_prompt(args, mode="editor")
    for i in range(len(generated_feedback)):
        data[i]['feedback'] = generated_feedback[i]

    all_model_data = format_model_input(prompt, data)

    return all_model_data 

def evaluate_result(input_data):
    reconstruct_prediction = []
    for data in input_data:
        reconstruct_prediction.append([data['prompt'] + '\n' + data['edited_code']])
    
    humaneval_fix = load_dataset("bigcode/humanevalpack")["test"]
    code_metric = load("Muennighoff/code_eval_octopack")
    language = "python"
    timeout = 1
    references = [d["test"] for d in humaneval_fix]
    results, logs = code_metric.compute(
        references=references,
        predictions=reconstruct_prediction,
        language=language,
        timeout=timeout,
        num_workers=4,
    )
    
    assert len(input_data) == len(logs)
    for i in range(len(logs)):
        input_data[i]['result'] = logs[i][0][1]
    
    return_dict = {'score': results, 'result': input_data}
    return return_dict
            
            


async def main(args):
    os.makedirs(args.save_dir, exist_ok=True)
    if args.feedback_gen == 'True':
        print(f"model_name: {args.critic_model_name}")
        if 'gpt' in args.critic_model_name:
            critic_llm = OpenAIModel(model_name=args.critic_model_name,
                            mode="critic",
                            temperature=args.temperature, 
                            top_p=args.top_p,
                            frequency_penalty=args.frequency_penalty)
        else:
            critic_llm = VllmModel(model_name=args.critic_model_name,
                            mode="critic",
                            port=args.critic_port,
                            url=args.critic_url,
                            temperature=args.temperature, 
                            top_p=args.top_p,
                            frequency_penalty=args.frequency_penalty)

    if 'gpt' in args.editor_model_name:
        editor_llm = OpenAIModel(model_name=args.editor_model_name,
                        mode="editor",
                        temperature=args.temperature, 
                        top_p=args.top_p,
                        frequency_penalty=args.frequency_penalty)
    else:
        editor_llm = VllmModel(model_name=args.editor_model_name,
                        mode="editor",
                        port=args.editor_port,
                        url=args.editor_url,
                        temperature=0,
                        top_p=1.0,
                        frequency_penalty=args.frequency_penalty)
        
    
    
    all_model_data = load_data(args)

    error_msg_li = evaluate_solutions(all_model_data, 'buggy_solution', 'example_test')
    
    generated_feedback = []
    for i in range(len(all_model_data)):
        assert error_msg_li[i]['task_id'] == all_model_data[i]['task_id']
        all_model_data[i]['error_msg'] = error_msg_li[i]['error_msg']
        generated_feedback.append('')
    
    
    if args.feedback_gen == 'True':
        #prepare feedback generation input 
        feedback_generation_model_input = prepare_feedback_generation_input(args, all_model_data)
        # pprint(feedback_generation_model_input[0])
        
        generated_feedback = await critic_llm.generate_concurrently(feedback_generation_model_input)

    # prepare code editing input
    code_editing_model_input = prepare_code_editing_input(args, all_model_data, generated_feedback)
    # pprint(code_editing_model_input[0])
    generated_code = await editor_llm.generate_concurrently(code_editing_model_input)
    raw_output = copy.deepcopy(generated_code)
    
    for k in range(len(generated_code)):
        splitted = extract_python_code(generated_code[k])
        generated_code[k] = splitted
        # if '```python' in generated_code[k]:
        #     splitted = generated_code[k].split("```python")[1]
        #     generated_code[k] = splitted
    # pprint(generated_code[0])    
    
    data_to_save = []
    assert len(generated_code) == len(generated_feedback), "the number of generated code and the number of generated feedback is different"
    # add generation results (both feedback and edited code) to original data dictionary
    for i in range(len(generated_feedback)):
        cur_dict = {k:v for k,v in all_model_data[i].items()}
        cur_dict['feedback'] = generated_feedback[i]
        cur_dict['edited_code'] = generated_code[i]
        cur_dict['model_input'] = code_editing_model_input[i]
        cur_dict['feedback_readable'] = generated_feedback[i].split("\n")
        cur_dict['model_input_readable'] = cur_dict['model_input'].split("\n")
        cur_dict['raw_output'] = raw_output[i].split("\n")
        cur_dict['buggy_solution_readable'] = cur_dict['buggy_solution'].split("\n")
        cur_dict['canonical_solution_readable'] = cur_dict['canonical_solution'].split("\n")
        cur_dict['edited_code_readable'] = generated_code[i].split("\n")

        data_to_save.append(cur_dict)
    
  
    data_to_save_prev = copy.deepcopy(data_to_save)
    sample_result = evaluate_solutions(data_to_save, 'edited_code', 'example_test')
    
    return_dict = evaluate_result(data_to_save)  
    
    with open(args.save_dir + 'initial_response.json', "w", encoding='UTF-8') as f:
        json.dump(return_dict, f, indent=4, ensure_ascii=False)
        
    
    if args.is_iter == 'True':
        for iter in range(args.iter_no):
            for j in range(len(all_model_data)):
                assert sample_result[j]['task_id'] == all_model_data[j]['task_id']
                all_model_data[j]['error_msg'] = sample_result[j]['error_msg']
                all_model_data[j]['buggy_solution'] = generated_code[j]

            if args.feedback_gen == 'True':
                feedback_generation_model_input = prepare_feedback_generation_input(args, all_model_data)
                # pprint(feedback_generation_model_input[0])
                
                generated_feedback = await critic_llm.generate_concurrently(feedback_generation_model_input)

            # prepare code editing input
            code_editing_model_input = prepare_code_editing_input(args, all_model_data, generated_feedback)
            # pprint(code_editing_model_input[0])
            generated_code = await editor_llm.generate_concurrently(code_editing_model_input)
            raw_output = copy.deepcopy(generated_code)
            
            for k in range(len(generated_code)):
                splitted = extract_python_code(generated_code[k])
                generated_code[k] = splitted
                # if '```python' in generated_code[k]:
                #     splitted = generated_code[k].split("```python")[1]
                #     generated_code[k] = splitted
            # pprint(generated_code[0])    
            
            data_to_save = []
            assert len(generated_code) == len(generated_feedback), "the number of generated code and the number of generated feedback is different"
            # add generation results (both feedback and edited code) to original data dictionary
            for i in range(len(generated_feedback)):
                if sample_result[i]['error_msg'] != 'correct':
                    cur_dict = {k:v for k,v in all_model_data[i].items()}
                    cur_dict['feedback'] = generated_feedback[i]
                    cur_dict['edited_code'] = generated_code[i]
                    cur_dict['model_input'] = code_editing_model_input[i]
                    cur_dict['feedback_readable'] = generated_feedback[i].split("\n")
                    cur_dict['model_input_readable'] = cur_dict['model_input'].split("\n")
                    cur_dict['raw_output'] = raw_output[i].split("\n")
                    cur_dict['buggy_solution_readable'] = cur_dict['buggy_solution'].split("\n")
                    cur_dict['canonical_solution_readable'] = cur_dict['canonical_solution'].split("\n")
                    cur_dict['edited_code_readable'] = generated_code[i].split("\n")
                    
                else:
                    cur_dict = data_to_save_prev[i]

                data_to_save.append(cur_dict)
            
            data_to_save_prev = copy.deepcopy(data_to_save)
            sample_result = evaluate_solutions(data_to_save, 'edited_code', 'example_test')
            
            return_dict = evaluate_result(data_to_save)
            
            with open(args.save_dir + str(iter+1) + 'iter_response.json', "w", encoding='UTF-8') as f:
                json.dump(return_dict, f, indent=4, ensure_ascii=False)
                
        
if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))




