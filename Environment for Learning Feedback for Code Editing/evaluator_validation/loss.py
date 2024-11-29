import argparse
import json
import os
import csv
from collections import Counter
from tqdm import tqdm
import torch

def open_json_file(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def save_json_file(data, file_path):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

def calculate_pass_at_1(data):
    total_len = len(data)
    pass_at_1 = 0
    for d in tqdm(data):
        if d["feedback_score_pairs"][0][-1] == 1:
            pass_at_1 += 1
    return pass_at_1 / total_len

def calculate_mse_loss(args, data, label):
    total_len = len(data)
    mse_sum = 0
    for i in tqdm(range(len(data))):
        if args.eval_type == "edit":
            sub = data[i]["feedback_score_pairs"][0][-1] - label[i]
        else:
            if "False" in data[i]["response"]:
                sub = 0 - label[i]
            else:
                sub = 1 - label[i]
        mse_sum = mse_sum + sub**2
    
    return mse_sum/total_len

def calculate_ce_loss(correct_data, wrong_data):
    correct_score_li = []
    for d in tqdm(correct_data):
        correct_score_li.append(float(d["feedback_score_pairs"][0][-1]))
    
    wrong_score_li = []
    for d in tqdm(wrong_data):
        wrong_score_li.append(float(d["feedback_score_pairs"][0][-1]))

    score = torch.tensor([correct_score_li, wrong_score_li])
    target = torch.tensor([1.0,0.0])

    loss = torch.nn.functional.cross_entropy(score, target.long())
    
    return loss.item()



def main(args):
    result_dir = args.result_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    results = {}

    for directory in os.listdir(result_dir):
        directory_path = os.path.join(result_dir, directory)
        if not os.path.isdir(directory_path):
            continue

        correct_feedback_file_path = None
        wrong_feedback_file_path = None

        for file in os.listdir(directory_path):
            if "correct" in file and file.endswith(".json"):
                correct_feedback_file_path = os.path.join(directory_path, file)
            elif "wrong" in file and file.endswith(".json"):
                wrong_feedback_file_path = os.path.join(directory_path, file)
        
        if correct_feedback_file_path and wrong_feedback_file_path:
            correct_feedback_file = open_json_file(correct_feedback_file_path)
            wrong_feedback_file = open_json_file(wrong_feedback_file_path)
            label = []
            for i in correct_feedback_file:
                label.append(1.0)
            for j in wrong_feedback_file:
                label.append(0.0)
            data = correct_feedback_file + wrong_feedback_file

            mse = calculate_mse_loss(args, data, label)
            if args.eval_type == "edit":
                ce = calculate_ce_loss(correct_feedback_file, wrong_feedback_file)
            else:
                ce = None
            results[directory] = {
                "MSE_loss": mse,
                "CE_loss": ce,
            }
        else:
            print(f"Could not find correct or wrong feedback file for {directory}")

    output_json = os.path.join(output_dir, f"loss.json")
    save_json_file(results, output_json)
    print(f"Results saved in {output_json}.")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--result_dir", type=str, required=True)
    parser.add_argument("-o", "--output_dir", type=str, required=True)
    parser.add_argument("-t", "--eval_type", type=str, default="edit", help="edit or geval")
    args = parser.parse_args()
    
    main(args)
