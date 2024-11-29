#!/bin/bash

export PYTHONPATH="$PYTHONPATH:PATH"

datasets=(
  "data/dpoSamplingData/sampling_data_train.json"
  "data/dpoSamplingData/sampling_data_eval.json"
)

for dataset in "${datasets[@]}"
do
  python main.py \
    --do_generate \
    --dataset "$dataset" \
    --save_dir "code-editing/results/ours_dpo_sampling_data_$(basename "$dataset" .json)" \
    --model_port 8000 \
    --model_name "MODEL_NAME" \
    --model_url "http://localhost" \
    --reward_model_url "http://localhost" \
    --reward_model_name "MODEL_NAME" \
    --reward_model_port 8000 \
    --feedback_type "generated" \
    --sampling_num 10 
done


