import argparse
import asyncio
import json
import yaml
import os
import random

import gzip
import pickle

import numpy as np
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--prompt_key", type=str, default=None)
    parser.add_argument("--save_dir", type=str, required=True, help="It should be a NEW DIRECTORY. Please do not use an existing")
    parser.add_argument("--num_sample", type=int, default=None, help="If you want to test your code by sampling a small number of data, you can set this argument.")
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
    clinical_key = ['age', 'sex', 'educ', 'marriage', 'apoe', 'mmse']
    apoe_dict = {0: 'no', 1: 'one copy of', 2: 'two copies of'}
    q21_dict = {'No': 'no', 'Yes': ''}
    label_dict = {'Dementia': 'Dementia', 'MCI': 'Mild Cognitive Impairment', 'CN': 'Normal Cognition'}
    
    with gzip.open(data_path, "rb") as f:
        data = pickle.load(f)
    
    prompt_ipt = dict()
    for idx, key in enumerate(clinical_key):
        if key == 'apoe':
            prompt_ipt[key] = apoe_dict[data['clinical_info'][idx]]
            continue
        prompt_ipt[key] = data['clinical_info'][idx]    
    
    for idx, (key, value) in enumerate(data['qa_info'].items()):
        if idx == 21:
            prompt_ipt[f'q{idx}'] = q21_dict[value]
            continue
        prompt_ipt[f'q{idx}'] = value
    
    label = label_dict[data['label']]
    prompt_ipt['label'] = label
    
    model_input = prompt.format(**prompt_ipt)
    return model_input, label

def load_and_prepare_data(args):
    prompt = load_prompt(args)
    all_model_inputs, all_labels = [], []
    print("Preparing model inputs...")
    for data_path in tqdm(os.listdir(args.input_path)):
        model_input, label = prepare_model_input(prompt, os.path.join(args.input_path, data_path))
        all_model_inputs.append(model_input)
        all_labels.append(label)
    return all_model_inputs, all_labels


def main(args):
    all_model_inputs, all_labels = load_and_prepare_data(args)
    
    save_data = []
    for idx, (model_input, label) in enumerate(zip(all_model_inputs, all_labels)):
        temp = dict()
        temp['id'] = idx
        temp['patient_data'] = model_input
        temp['label'] = label
        save_data.append(temp)
    
    with open(os.path.join(args.save_dir, "data.json"), "w", encoding="UTF-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=4)
        
if __name__ == "__main__":
    args = parse_args()
    main(args)
