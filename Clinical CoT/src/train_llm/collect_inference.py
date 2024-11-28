import json
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str) 
parser.add_argument("--file_name")
parser.add_argument("--output_file")
args = parser.parse_args()


files = [os.path.join(args.input_dir,fn) for fn in os.listdir(args.input_dir) if args.file_name in fn]


aggregated = []
for fn in files:
    data = json.load(open(fn, "r"))
    aggregated.extend(data)


with open(args.output_file, 'w') as f:
    json.dump(aggregated, f, indent=4)

    
