import argparse
import json
import os
import csv
from collections import Counter
from tqdm import tqdm


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
            correct_feedback_pass_at_1 = calculate_pass_at_1(correct_feedback_file)
            wrong_feedback_pass_at_1 = calculate_pass_at_1(wrong_feedback_file)
            results[directory] = {
                "correct_feedback_pass_at_1": correct_feedback_pass_at_1,
                "wrong_feedback_pass_at_1": wrong_feedback_pass_at_1
            }
        else:
            print(f"Could not find correct or wrong feedback file for {directory}")

    output_json = os.path.join(output_dir, f"pass_at_1_results.json")
    save_json_file(results, output_json)
    print(f"Results saved in {output_json}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--result_dir", type=str, required=True)
    parser.add_argument("-o", "--output_dir", type=str, required=True)
    args = parser.parse_args()
    
    main(args)
