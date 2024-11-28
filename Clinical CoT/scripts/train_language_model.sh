#!/bin/bash

 deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port 30000  src/train_llm/train_clm.py \
    --model_name_or_path /model/name/or/llama2_7B/dir \
    --deepspeed ds_config_zero2.json \
    --output_dir /your/checkpoint/dir/for/language/student/model \
    --dataset  /dataset/dir/path \
    --do_train True \
    --do_eval True --do_mmlu_eval False \
    --source_max_len 600 \
    --target_max_len 1024 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --logging_steps 10 \
    --max_steps 10000 \
    --save_strategy epoch \
    --data_seed 42 \
    --save_total_limit 40 \
    --evaluation_strategy epoch \
    --eval_dataset_size 1024 \
    --max_eval_samples 1000 \
    --optim paged_adamw_32bit \
    --report_to wandb --bits 16 --bf16 False 
