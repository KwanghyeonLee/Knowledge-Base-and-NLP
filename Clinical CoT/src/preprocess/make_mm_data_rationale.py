import numpy as np
import json
import os
from tqdm import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--rationale_path", type=str, default=None)
    parser.add_argument("--save_dir", type=str, required=True)
    args = parser.parse_args()

    return args

def main(args):
    
    # Open rationale file
    with open(args.rationale_path, "r", encoding="UTF-8") as f:
        rationale = json.load(f)
        if 'data' in rationale:
            rationale = rationale['data']
    
    # Iterate data dir
    for file in tqdm(os.listdir(args.input_path)):
        data = np.load(os.path.join(args.input_path, file), allow_pickle=True)
        temp = dict()
        for r in rationale:
            if data['patient_data'] == r['patient_data']:
                temp['id'] = data['id']
                temp['patient_data'] = data['patient_data']
                temp['image_mr']  = np.expand_dims(data['image_mr'], axis=0)
                temp['image_parcel'] = data['image_parcel']
                temp['input'] = r['input']
                temp['label'] = r['label']
                temp['org_label'] = r['org_label']
                np.savez_compressed(os.path.join(args.save_dir, f"{data['id']}.npz"), **temp)
                break
            
if __name__ == "__main__":
    args = parse_args()
    main(args)
                