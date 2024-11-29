#!/bin/bash
export OPENAI_API_KEY="KEY"

export PYTHONPATH="$PYTHONPATH:PATH"

python main.py \
  --reward_model_name MODEL_NAME \
  --iter_data_size 500 \
  --save_dir save_dir
