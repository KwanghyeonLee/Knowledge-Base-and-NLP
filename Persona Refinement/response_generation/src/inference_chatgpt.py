import argparse
import asyncio
import json
import yaml
import os
import random
from copy import deepcopy
import gzip
import pickle
import signal
import aiohttp

import numpy as np
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAIChat, OpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

import tiktoken


TOTAL_COST = 0  

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str)
    parser.add_argument("--org_id", type=str)
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--prompt_key", type=str, default=None)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--num_sample", type=int, default=None)
    parser.add_argument("--ordered_sample", type=int, default=None)
    parser.add_argument("--ordered_session", type=int, default=None)
    parser.add_argument("--no_memory", action="store_true")
    parser.add_argument("--task", type=str, choices=["chat"], default="chat")
    parser.add_argument("--window_size", type=int)
    parser.add_argument("--call_amount", type=int, default=300)
    parser.add_argument("--use_retrieval", action="store_true")

    # parser.add_argument("--min_context_len", type=int, default=0)
    args = parser.parse_args()
    if args.num_sample:
        args.save_dir = args.save_dir + f"_sample{args.num_sample}"
    return args

def set_openai_api(api_key, org_id):
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["OPENAI_ORGANIZATION"] = org_id
    print(f"Set OpenAI API Key and Organization.")

def load_prompt(args):
    with open(args.prompt, "r", encoding="UTF-8") as f:
        prompt = yaml.load(f, Loader=yaml.FullLoader)[args.prompt_key]
    return prompt


def prepare_model_input(prompt:str, data_path:str, args):
    global TOTAL_COST
    '''
        input : prompt, data_path (str)
        output : all_model_data (list of dict)
    '''
    with open(data_path, "r", encoding="UTF-8") as f:
        data = json.load(f)

    if args.ordered_sample:
        if args.ordered_session:
            data = [item for item in data if f"-{args.ordered_session}-" in item["id"] and item["episode_id"] <= args.ordered_sample-1]
        else:
            data = [item for item in data if item["episode_id"] <= args.ordered_sample-1]
            

    all_model_data = []
    
    for d in data:
        input_temp = dict()
        

        if args.use_retrieval:

            input_temp['id'], input_temp['label'], input_temp['memory'], input_temp['dialog'], input_temp['response_que'] = d['id'], d['label'], d['memory_retrieved'], d['dialog'], d['response_que']
            if args.no_memory:
                facts_a = "None."
                facts_b = "None."
            else:
                facts_a = "\n".join(d["memory_retrieved"][0]) if len(d["memory_retrieved"][0]) > 0 else "None."
                facts_b = "\n".join(d["memory_retrieved"][1]) if len(d["memory_retrieved"][1]) > 0 else "None."
            input_temp['model_input'] = prompt.format(**{
                "facts_a": facts_a,
                "facts_b": facts_b,
                "dialogue_context": "\n".join(d["dialog"]) if len(d["dialog"]) > 0 else "None.",
                "response_que": d["response_que"]
        
            })
            all_model_data.append(input_temp)

        else:
            input_temp['id'], input_temp['label'], input_temp['memory'], input_temp['dialog'], input_temp['response_que'] = d['id'], d['label'], d['memory'], d['dialog'], d['response_que']
            if args.no_memory:
                facts_a = "None."
                facts_b = "None."
            else:
                
                facts_a = "\n".join(d["memory"][0]) if len(d["memory"][0]) > 0 else "None."
                facts_b = "\n".join(d["memory"][1]) if len(d["memory"][1]) > 0 else "None."
            input_temp['model_input'] = prompt.format(**{
                "facts_a": facts_a,
                "facts_b": facts_b,
                "dialogue_context": "\n".join(d["dialog"]) if len(d["dialog"]) > 0 else "None.",
                "response_que": d["response_que"]
        
            })
            all_model_data.append(input_temp)


    return all_model_data

def load_and_prepare_data(args):
    prompt = load_prompt(args)
    
    print("Preparing model inputs...")
    all_model_data = prepare_model_input(
        prompt.rstrip('\n'), args.input_path, args)
    return all_model_data


def sample_indices(all_model_inputs, num_sample):
    random.seed(0)
    cand_indices = list(range(len(all_model_inputs)))
    sampled_indices = random.sample(cand_indices, num_sample)
    return sampled_indices



def filter_data(all_model_data, num_sample):
    
    if num_sample:
        sampled_indices = sample_indices(all_model_data, num_sample)
        all_model_data = [all_model_data[i] for i in sampled_indices]

    return all_model_data



async def async_generate(llm, model_data, idx, save_dir, args):
    global TOTAL_COST

    if os.path.exists(os.path.join(save_dir, f"{idx}.json")):
        with open(os.path.join(save_dir, f"{idx}.json"), "r", encoding='UTF-8') as f:
            result = json.load(f)
        
        return result
    
    message = model_data['model_input']
    
    system_message = SystemMessage(content=message)
    
    async with aiohttp.client.ClientSession() as session:
        while True:
            try:
                response = await asyncio.wait_for(llm.agenerate([[system_message]]), timeout=120)

                token_used = response.llm_output['token_usage']['total_tokens']

                TOTAL_COST += token_used / 1000 * 0.002 #)  # gpt-3.5-turbo
                # TOTAL_COST += token_used / 1000 * 0.06  # gpt-4
                
                # print(idx, TOTAL_COST)
                break

            except Exception as e:
                print(f"Exception occurred: {str(e)}")
                response = None
   
    result = deepcopy(model_data)
    result['prediction'] = response.generations[0][0].text

    with open(os.path.join(save_dir, f"{idx}.json"), "w", encoding='UTF-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    return result


async def generate_concurrently(all_model_data, start_idx, save_dir):
    llm = ChatOpenAI(model_name='gpt-3.5-turbo',  # 'gpt-3.5-turbo' or 'gpt-4'
                     temperature=0.7, max_tokens=300, max_retries=100,
                     request_timeout=60)
    tasks = [async_generate(llm, model_data, i+start_idx, save_dir, args) for i, model_data in enumerate(all_model_data)]

    return await tqdm_asyncio.gather(*tasks)


async def main(args):
    global TOTAL_COST
    total_result_path = args.save_dir + ".json"

    if os.path.exists(total_result_path):
        return
    else:        
        all_model_data = load_and_prepare_data(args)
        all_model_data = filter_data(all_model_data, args.num_sample)
    
        if os.path.exists(args.save_dir):
            print("The save_dir already exists. Please change the save_dir.")

        os.makedirs(args.save_dir, exist_ok=True)
        all_results = []
        if len(all_model_data) > args.call_amount:
            for start_idx in tqdm(range(0, len(all_model_data), args.call_amount), desc="Processing", unit="item"):
                cur_model_data = all_model_data[start_idx:start_idx + args.call_amount]
                all_results.extend(await generate_concurrently(cur_model_data, start_idx, args.save_dir))
                        
        else:
            all_results = await generate_concurrently(all_model_data, 0, args.save_dir)
            
        with open(os.path.join(total_result_path), "w", encoding='UTF-8') as f:
            json.dump(all_results, f, indent=4, ensure_ascii=False)

        
if __name__ == "__main__":
    args = parse_args()
    set_openai_api(args.api_key, args.org_id)
    while True:
        try:
            asyncio.run(main(args))
            exit()
        except Exception as e:
            print(e)
            pass

        asyncio.sleep(10)