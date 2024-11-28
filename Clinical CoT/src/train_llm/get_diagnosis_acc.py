import json
import os
import re

from tqdm import tqdm
import argparse

answer_dict = {
    'Normal Cognition': 'C',
    'Mild Cognitive Impairment': 'B',
    'Dementia': 'A'
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--label_name", type=str, default="org_label")
    args = parser.parse_args()
    return args

def get_answer(answer):
    ans = re.findall(r'\([A-D]\)', answer)
    ans = [a[1:-1] for a in ans]
    if len(ans) == 0:
        return None
    return ans[0]

def main(args):
    # open file
    with open(args.input_path, "r", encoding="UTF-8") as f:
        data = json.load(f)
        if 'data' in data:
            data = data['data']
    
    miss = 0
    new_data = []
    for d in data:
        if 'prediction' not in d:
            miss += 1
            continue
        new_data.append(d)
    data = new_data
    
    total_CN = len([d for d in data if d[args.label_name] == "Normal Cognition"])
    total_MCI = len([d for d in data if d[args.label_name] == "Mild Cognitive Impairment"])
    total_Dementia = len([d for d in data if d[args.label_name] == "Dementia"])
    
    # for precision
    total_CN_pred = len([d for d in data if get_answer(d["prediction"]) == answer_dict["Normal Cognition"]])
    total_MCI_pred = len([d for d in data if get_answer(d["prediction"]) == answer_dict["Mild Cognitive Impairment"]])
    total_Dementia_pred = len([d for d in data if get_answer(d["prediction"]) == answer_dict["Dementia"]])
    # get accuracy
    correct = 0
    correct_CN = 0
    correct_MCI = 0
    correct_Dementia = 0
 
    for d in tqdm(data):
        if get_answer(d['prediction']) == answer_dict[d[args.label_name]]:
            correct += 1
            if d[args.label_name] == 'Normal Cognition':
                correct_CN += 1
            elif d[args.label_name] == 'Mild Cognitive Impairment':
                correct_MCI += 1
            elif d[args.label_name] == 'Dementia':
                correct_Dementia += 1
    
    # print accuracy
    print(f"Accuracy: {correct/len(data)}")
    print(f"CN recall: {correct_CN/total_CN}")
    print(f"MCI recall: {correct_MCI/total_MCI}")
    print(f"AD recall: {correct_Dementia/total_Dementia}")
    print("---------------------------")
    print(f"CN precision: {correct_CN/total_CN_pred}")
    print(f"MCI precision: {correct_MCI/total_MCI_pred}")
    print(f"AD precision: {correct_Dementia/total_Dementia_pred}")
    print("---------------------------")
    print(f"Misssing: {miss}")    
    metrics = {
        "total_acc": correct/len(data),
        "NC_recall": correct_CN/total_CN,
        "MCI_recall": correct_MCI/total_MCI,
        "AD_recall": correct_Dementia/total_Dementia,
        "NC_precision": correct_CN/total_CN_pred,
        "MCI_precision": correct_MCI/total_MCI_pred,
        "AD_precision": correct_Dementia/total_Dementia_pred
    }
    
    save_path = args.input_path.replace(".json", "_metrics.json")
    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    args = parse_args()
    main(args)