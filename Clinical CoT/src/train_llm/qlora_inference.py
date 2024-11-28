import torch
from torch.utils.data import DataLoader
from transformers import BitsAndBytesConfig
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM
from peft import PeftModel
from transformers import pipeline
import transformers
# import deepspeed
import torch
import os
import argparse
from transformers import AutoTokenizer
import json
from tqdm import tqdm
from torch.utils.data import Dataset
parser = argparse.ArgumentParser()

parser.add_argument("--input_file", type=str)
parser.add_argument("--output_file", type=str)
parser.add_argument("--batch_size", type=int)
parser.add_argument("--model_name", type=str)
parser.add_argument("--max_length", type=int)
parser.add_argument("--is_clm",  dest="is_clm", action="store_true")
parser.add_argument("--no_clm",  dest="is_clm", action="store_false")
parser.add_argument("--split", type=int, default=None)
parser.add_argument("--split_idx", type=int)
parser.add_argument("--do_sample", dest="do_sample", action="store_true")
parser.add_argument("--no-do_sample", dest="do_sample", action="store_false")
parser.add_argument("--top_k", type=float)
parser.add_argument("--top_p", type=float)
parser.add_argument("--temperature", type=float)
# parser = deepspeed.add_config_arguments(parser)

args = parser.parse_args()
if args.is_clm:
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, max_length=1024,truncate_size="left", truncation=True, padding_side="left") 
else:
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, max_length=1024,truncate_size="left", truncation=True) 


def load_data():
    with open(args.input_file, "r") as f:
        data = json.load(f)
        if "data" in data:
            data = data["data"]
    
    return data


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
print("Loading model")
try:
    model_4bit = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, quantization_config=bnb_config, device_map="auto")
except:
    model_4bit = AutoModelForCausalLM.from_pretrained(args.model_name, quantization_config=bnb_config, device_map="auto")

print("Loading data")
data = load_data()
if args.split:
    len_split = int(len(data)/args.split)+1
    data = data[len_split*args.split_idx:len_split*(args.split_idx+1)]
batch_size = args.batch_size

class ListDataset(Dataset):
    def __init__(self, original_list):
        self.original_list = original_list

    def __len__(self):
        return len(self.original_list)

    def __getitem__(self, i):
        return self.original_list[i]


my_dataset = ListDataset([d['input'] for d in data])
dataloader = DataLoader(dataset=my_dataset, batch_size=batch_size, shuffle=False)
device = "cuda"

print("Start inference")
batch_outputs = []
for batch in tqdm(dataloader):
    input_seq = batch
    # tokenizer.pad_token = tokenizer.eos_token
    tokenized_input = tokenizer.batch_encode_plus(input_seq,return_tensors="pt", truncation=True, padding="longest", max_length=1024).to(device)
    outputs = model_4bit.generate(tokenized_input["input_ids"], max_new_tokens=args.max_length, top_p=args.top_p, top_k=args.top_k, do_sample=args.do_sample, temperature=args.temperature)
    output_seq = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    decoded_input = tokenizer.batch_decode(tokenized_input["input_ids"], skip_special_tokens=True)
    if args.is_clm:
        for i in range(len(output_seq)):
            output_seq[i] = output_seq[i][len(decoded_input[i]):]
    batch_outputs.extend(output_seq)

save_list = []
for i in range(len(batch_outputs)):
    dict_to_save = {"prediction": batch_outputs[i]}
    for k,v in data[i].items():
        dict_to_save[k] = v
    save_list.append(dict_to_save)
    
output_file_name = args.output_file
if args.split:
    output_file_name = output_file_name.replace(".json", f"_split_{args.split_idx}_{args.split}.json")
with open(output_file_name,"w") as f:
    json.dump(save_list, f, indent=4)
