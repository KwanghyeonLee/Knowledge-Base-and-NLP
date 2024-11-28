#!/bin/bash

echo "Start running LLaMA2 7b"
for split_idx in {0..7};do
    log_file="logs/process_${split_idx}.log"
    CUDA_VISIBLE_DEVICES=$split_idx python src/train_llm/qlora_inference.py \
    --model_name model/checkpoint/dir \
    --input_file input/data/json/file/path \
    --output_file output/data/json/file/path \
    --batch_size=8 --max_length=1024 --split=8 --split_idx=$split_idx --is_clm > "$log_file" 2>&1 &
done

wait

echo "Finished running LLaMA2 7b and Collecting inference files"

python src/train_llm/collect_inference.py \
    --file_name test \
    --input_dir prev/output/data/dir/path \
    --output_file output/data/json/file/path 

echo "Finished collecting inference files for LLaMA7b"

echo "Get metrics for LLaMA7b"
python src/train_llm/get_diagnosis_acc.py \
    --input_path output/data/json/file/path 

