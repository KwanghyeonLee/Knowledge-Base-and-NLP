import argparse
import json
import os
import csv
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import matthews_corrcoef
from collections import Counter

def process_files(correct_feedback_file, wrong_feedback_file, evaluation_type):
    ref_arr = []
    pred_arr = []
    error_count = 0

    with open(correct_feedback_file, "r") as f:
        correct_eval_data_raw = json.load(f)

    with open(wrong_feedback_file, "r") as f:
        wrong_eval_data_raw = json.load(f)

    if evaluation_type == "geval":
        # Process correct evaluation data
        for d in correct_eval_data_raw:
            try:
                pred_arr.append(int("True" == d['response'][0]))
                ref_arr.append(1)
            except Exception as e:
                error_count += 1
                print(f"Error processing correct eval data: {e}")

        # Process wrong evaluation data
        for d in wrong_eval_data_raw:
            try:
                pred_arr.append(int("True" == d['response'][0]))
                ref_arr.append(0)
            except Exception as e:
                error_count += 1
                print(f"Error processing wrong eval data: {e}")
    else:
        # Process correct evaluation data
        for d in correct_eval_data_raw:
            try:
                pred_arr.append(int(d['feedback_score_pairs'][0][1]))
                ref_arr.append(1)
            except Exception as e:
                error_count += 1
                print(f"Error processing correct eval data: {e}")

        # Process wrong evaluation data
        for d in wrong_eval_data_raw:
            try:
                pred_arr.append(int(d['feedback_score_pairs'][0][1]))
                ref_arr.append(0)
            except Exception as e:
                error_count += 1
                print(f"Error processing wrong eval data: {e}")

    print("Distribution of ref_arr:", dict(Counter(ref_arr)))
    print("Distribution of pred_arr:", dict(Counter(pred_arr)))

    if len(set(ref_arr)) == 1 or len(set(pred_arr)) == 1:
        correlation_results = {
            'pearson_corr': float('nan'),
            'pearson_p_value': float('nan'),
            'spearman_corr': float('nan'),
            'spearman_p_value': float('nan'),
            'kendall_corr': float('nan'),
            'kendall_p_value': float('nan'),
            'mcc': float('nan')
        }
    else:
        pearson_corr, pearson_p_value = pearsonr(pred_arr, ref_arr)
        spearman_corr, spearman_p_value = spearmanr(pred_arr, ref_arr)
        kendall_corr, kendall_p_value = kendalltau(pred_arr, ref_arr)
        mcc = matthews_corrcoef(ref_arr, pred_arr)

        correlation_results = {
            'pearson_corr': pearson_corr,
            'pearson_p_value': pearson_p_value,
            'spearman_corr': spearman_corr,
            'spearman_p_value': spearman_p_value,
            'kendall_corr': kendall_corr,
            'kendall_p_value': kendall_p_value,
            'mcc': mcc
        }

    return correlation_results, error_count



def save_results(results, output_csv, output_json):
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['file', 'pearson_corr', 'pearson_p_value', 'spearman_corr', 'spearman_p_value', 'kendall_corr', 'kendall_p_value', 'mcc', 'error_count']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for filename, data in results.items():
            row = {'file': filename}
            row.update(data['results'])
            row['error_count'] = data['error_count']
            writer.writerow(row)
    
    with open(output_json, 'w') as jsonfile:
        json.dump(results, jsonfile, indent=4)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--type", type=str, choices=["geval", "editing"], required=True)
    parser.add_argument("-i", "--result_dir", type=str, required=True)
    parser.add_argument("-o", "--output_dir", type=str, required=True)
    args = parser.parse_args()

    evaluation_type = args.type
    result_dir = args.result_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    results = {}
    error_count = 0

    for directory in os.listdir(result_dir):
        directory_path = os.path.join(result_dir, directory)
        if not os.path.isdir(directory_path):
            continue

        correct_feedback_file = None
        wrong_feedback_file = None

        for file in os.listdir(directory_path):
            if "correct" in file and file.endswith(".json"):
                correct_feedback_file = os.path.join(directory_path, file)
            elif "wrong" in file and file.endswith(".json"):
                wrong_feedback_file = os.path.join(directory_path, file)
        
        if correct_feedback_file and wrong_feedback_file:
            result, errors = process_files(correct_feedback_file, wrong_feedback_file, evaluation_type)
            results[directory] = {'results': result, 'error_count': errors}
            error_count += errors

    output_csv = os.path.join(output_dir, f"correlation_results.csv")
    output_json = os.path.join(output_dir, f"correlation_results.json")
    save_results(results, output_csv, output_json)
    print(f"Results saved in {output_csv} and {output_json}. Total errors encountered: {error_count}")

if __name__ == "__main__":
    main()
