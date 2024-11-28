import os
import argparse
import json
import numpy as np
import torch
from tqdm import tqdm
import nltk
#nltk.download('punkt')
from nltk.tokenize import word_tokenize
import evaluate

your_path = ''

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="result/response_generation/direct_rg_total_results.json")
    parser.add_argument("--nltk", action="store_true")
    parser.add_argument("--task", choices=["direct", "rationale"], default="direct")
    parser.add_argument("--no_bertscore", action="store_true")
    parser.add_argument("--all_session_metric", action="store_true")
    parser.add_argument("--session_id", type=int)
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--sheet_name", type=str, required=True)
    parser.add_argument("--memory_turn", action='store_true')
    parser.add_argument("--memory_turn_thr", type=float, default=0.4)

    args = parser.parse_args()
    return args

def load_data(args):
    if args.memory_turn:
        with open(f'{your_path}/Caffeine/data/msc_gold_helpfulness_both/memory_turn_{args.memory_turn_thr}.txt', 'r',encoding='utf-8') as f:
            memory_turn = f.read()
            memory_turn_list = memory_turn.split('\n')
        
        with open(args.input_path, "r", encoding="utf-8") as f:
            all_data = json.load(f)
        
        data = []
        
        for i in all_data:
            
            if i['id'] in memory_turn_list:
                data.append(i)
                
            
    else:    
        with open(args.input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
    return data

def get_data(args, data):
    return_data = {
        "all": {
            "total_list": [],
            "prediction_list": [],
            "reference_list": []
        },
        "1": { # session 2 (1-indexed)
            "total_list": [],
            "prediction_list": [],
            "reference_list": []
        },
        "2": { # session 3 (1-indexed)
            "total_list": [],
            "prediction_list": [],
            "reference_list": []
        },
        "3": { # session 4 (1-indexed)
            "total_list": [],
            "prediction_list": [],
            "reference_list": []
        },
        "4": { # session 5 (1-indexed)
            "total_list": [],
            "prediction_list": [],
            "reference_list": []
        },
    }
    for d in tqdm(data):
        if args.all_session_metric:
            session_id = d["id"].split("-")[1]
        if args.session_id :
            session_id = str(args.session_id)
        
        # parse the model prediction
        if args.task == "direct":
            model_pred = d["prediction"].strip()
        else:
            model_pred = d["prediction"].split(":")[-1].strip()
        
        # append to total_list 
        temp_dict = {
            "id": d["id"],
            "dialog": d["dialog"],
            "predictions": model_pred,
            "references": d["label"]
        }
        if "memory" in d:
            temp_dict["memory"] = d["memory"]
        
        return_data['all']['total_list'].append(temp_dict)
        return_data[session_id]['total_list'].append(temp_dict)
            
        # append to prediction_list
        return_data['all']['prediction_list'].append(model_pred)
        return_data[session_id]['prediction_list'].append(model_pred)
        
        # append to reference_list
        return_data['all']['reference_list'].append(d["label"])
        return_data[session_id]['reference_list'].append(d["label"])
            
    return return_data

def compute_metric(predictions: list, references: list):
    # load the metric
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")
    bertscore = evaluate.load("bertscore")
    
    # convert to lower case
    all_predictions = [pred.lower() for pred in predictions]
    all_targets = [target.lower() for target in references]
        
    all_metric = {}
    
    # compute rouge scores
    print("Computing rouge scores...")
    
    rouge_score = rouge.compute(predictions=all_predictions, references=all_targets)
    for k,v in rouge_score.items():
        all_metric[k] = str(round(v,4))

    # compute bleu scores
    print("Computing bleu scores...")
    for i in range(1,5):
        if args.nltk:
            bleu_score = bleu.compute(predictions=all_predictions, references=all_targets, tokenizer= word_tokenize, max_order=i)['bleu']
        else:
            bleu_score = bleu.compute(predictions=all_predictions, references=all_targets, max_order=i)['bleu']
        all_metric[f"bleu{i}"] = str(round(bleu_score,4))

    # compute bertscore
    # check is cuda available
    if not torch.cuda.is_available():
        print("No cuda available, skipping bertscore...")
        args.no_bertscore = True
        
    if not args.no_bertscore:
        print("Computing bertscore...")
        gpu_list = [0,1,2,3,4,5,6,7]
        bertscore_score = round(np.mean(bertscore.compute(predictions=all_predictions, references=all_targets, lang="en", device=f'cuda:{get_least_used_gpu(gpu_list)}')['f1']),4)
        all_metric["bertscore"] = str(bertscore_score)
    
    return all_metric

def main(args):
    print("Loading data...")
    base_data = load_data(args)
    
    print("Getting data...")
    result_dict = get_data(args, base_data)

    print("Computing metrics...") 
    if args.all_session_metric :
        session1_metric = compute_metric(result_dict['1']['prediction_list'], result_dict['1']['reference_list'])
        session2_metric = compute_metric(result_dict['2']['prediction_list'], result_dict['2']['reference_list'])
        session3_metric = compute_metric(result_dict['3']['prediction_list'], result_dict['3']['reference_list'])
        session4_metric = compute_metric(result_dict['4']['prediction_list'], result_dict['4']['reference_list'])
        all_metric = compute_metric(result_dict['all']['prediction_list'], result_dict['all']['reference_list'])

    else:
        if args.session_id == 1:
            session1_metric = compute_metric(result_dict['1']['prediction_list'], result_dict['1']['reference_list'])
        elif args.session_id == 2:
            session2_metric = compute_metric(result_dict['2']['prediction_list'], result_dict['2']['reference_list'])
        elif args.session_id == 3:
            session3_metric = compute_metric(result_dict['3']['prediction_list'], result_dict['3']['reference_list'])
        elif args.session_id == 4:
            session4_metric = compute_metric(result_dict['4']['prediction_list'], result_dict['4']['reference_list'])
    
    print("Saving metrics...")
    save_dir = args.input_path.replace(".json","_metrics.csv")
    
    if args.nltk:
        save_dir = save_dir.replace(".csv","_nltk.csv")
        
    if args.memory_turn:
        save_dir = save_dir.split('.')[0] + f"_memory_turn_{args.memory_turn_thr}" + '.' + save_dir.split('.')[1]
        args.method = args.method + f"_memory_turn_{args.memory_turn_thr}"
    
    if os.path.exists(save_dir):
        return
    
    with open(save_dir,"w") as f:
        if args.all_session_metric :
            f.write(",".join(list(all_metric.keys()))+"\n")
            f.write(",".join(list(session1_metric.values())) + "\n")
            f.write(",".join(list(session2_metric.values())) + "\n")  
            f.write(",".join(list(session3_metric.values())) + "\n")
            f.write(",".join(list(session4_metric.values())) + "\n")
            return

        else:

            if args.session_id == 1:
                f.write(",".join(list(session1_metric.values())) + "\n")
            elif args.session_id == 2:
                f.write(",".join(list(session2_metric.values())) + "\n")
            elif args.session_id == 3:
                f.write(",".join(list(session3_metric.values())) + "\n")
            elif args.session_id == 4:
                f.write(",".join(list(session4_metric.values())) + "\n")
        
if __name__ == "__main__":
    args = get_args()
    main(args)