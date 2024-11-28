import os

import sys
your_path = ''
sys.path.append(f"{your_path}/Caffeine/response_generation")

import argparse
import json
import numpy as np
from tqdm import tqdm
import torch
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    pipeline,
    EvalPrediction,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
import faiss
import time
import multiprocessing as mp
import ast

from src.contriever import Contriever
from utils.gpu_check import get_least_used_gpu, get_least_used_gpu_multiple


def Retrieve(document_list:list, query_list: list, args, retriever):
    # Check if the input is valid
    if not document_list or not query_list or len(query_list) == 0:
        return [["None."],["None."]]
    
    # set document and query
    document_input = []
    for document in document_list:
        for d in document:
            document_input.append(d)
    
    query_input = "\n".join(query_list)
    
    # set retriever
    contriever, tokenizer, device = retriever
    
    try:
        document_ipt = tokenizer(document_input, padding=True, truncation=True, return_tensors="pt").to(device)
    except:
        print(document_input, document_list)
        # exit()
        return [[],[]]
    
    contriever.eval()
    
    with torch.no_grad():
        document_embed = contriever(**document_ipt).cpu().numpy()
    
    query_ipt = tokenizer(query_input, padding=True, truncation=True, return_tensors="pt").to(device)
    
    with torch.no_grad():
        query_embed = contriever(**query_ipt).cpu().numpy()

    document_embed = np.array(document_embed)
    document_embed = np.float32(document_embed)
    dimension = document_embed.shape[1]
    index = faiss.IndexFlatIP(dimension)    

    faiss.normalize_L2(document_embed)  
    index.add(document_embed)  
    
    query_embed = np.array(query_embed)
    query_embed = np.float32(query_embed)
    faiss.normalize_L2(query_embed)
    
    D,I = index.search(query_embed, args.top_k)
    
    return_list = [[],[]]
    for index in I[0]:
        if index < len(document_list[0]):
            return_list[0].append(document_input[index])
        else:
            return_list[1].append(document_input[index])
    return return_list  


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default=None, help="Input data path")
    parser.add_argument("--output_dir", type=str, default=None, help="Output data path")
    parser.add_argument("--retriever", choices=["openai", "contriever"], default="openai")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--org_id", type=str, default=None, help="OpenAI organization ID")
    parser.add_argument("--api_key", type=str, default=None, help="OpenAI API key")
    parser.add_argument('--gpu_list', type=str, default='')

    args = parser.parse_args()
    
    if args.org_id:
        os.environ["OPENAI_ORGANIZATION"] = args.org_id
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key
    return args

def model_process(model, chunk, argument, return_dict, idx):
    processed_data = []
    for d in tqdm(chunk):
        processed_data.append({
            'id': d['id'],
            'episode_id': d['episode_id'],
            'session_id': d['session_id'],
            'uttr_id': d['uttr_id'],
            'response_que': d['response_que'],
            'memory': d['memory'],
            'memory_retrieved': Retrieve(d['memory'], d['dialog'], argument, model) if len(d['dialog']) > 0 else d['memory'],
            'dialog': d['dialog'],
            'label': d['label'],
        })

    return_dict[idx] = processed_data

def parallel_retrieve(models, data_chunks, argument):
    manager = mp.Manager()
    return_dict = manager.dict()
    
    processes = []
    idx = 0
    for model, chunk in zip(models, data_chunks):
        p = mp.Process(target=model_process, args=(model, chunk, argument, return_dict, idx))
        p.start()
        processes.append(p)
        time.sleep(1)
        idx += 1

    for p in processes:
        p.join()
        
    return return_dict

def divide_list(lst, div):
    N = len(lst)
    base_length = N // div
    remainder = N % div

    parts = []
    start_idx = 0
    for i in range(div):
        if i < remainder:
            end_idx = start_idx + base_length + 1
        else:
            end_idx = start_idx + base_length
        parts.append(lst[start_idx:end_idx])
        start_idx = end_idx

    return parts

def main(args):
    os.makedirs(os.path.join(args.output_dir, str(args.top_k)), exist_ok=True)
    save_path = os.path.join(args.output_dir, str(args.top_k), args.input_path.split("/")[-1])

    if os.path.exists(save_path):
        return
    else:
        # load data
        print("Loading data...")
        with open(args.input_path, "r", encoding="UTF-8") as f:
            data = json.load(f)

        # set retriever
        print("Setting retriever...") 
        gpu_list = ast.literal_eval(args.gpu_list)
        div = 8
        gpu_idx = get_least_used_gpu_multiple(gpu_list, div)
        models = []
        for gpu_id in gpu_idx:
            device = f"cuda:{gpu_id}"
            contriever = Contriever.from_pretrained("facebook/contriever") 
            contriever = contriever.to(device)
            tokenizer = AutoTokenizer.from_pretrained("facebook/contriever", truncation_side='left')
            retriever = (contriever, tokenizer, device)
            
            models.append(retriever)

        print("Parallel Retrieving...")
        data_chunks = divide_list(data, div)
        result_tmp = parallel_retrieve(models, data_chunks, args)

        print("Sorting the Results...")
        new_data = []
        for j in range(len(gpu_idx)):
            processed_data = result_tmp[j]
            for d in processed_data:
                obs_dict = {
                    'id': d['id'],
                    'episode_id': d['episode_id'],
                    'session_id': d['session_id'],
                    'uttr_id': d['uttr_id'],
                    'response_que': d['response_que'],
                    'memory': d['memory'],
                    'memory_retrieved': d['memory_retrieved'],
                    'dialog': d['dialog'],
                    'label': d['label']
                }
                new_data.append(obs_dict)

        print(f"Retrieved {len(new_data)} data.")

        print("Saving data...")
        with open(save_path, "w", encoding="UTF-8") as f:
            json.dump(new_data, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    mp.set_start_method('spawn')
    args = get_args()
    main(args)