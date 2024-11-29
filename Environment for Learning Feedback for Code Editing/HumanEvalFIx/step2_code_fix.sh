#!/bin/bash
export PYTHONPATH="$PYTHONPATH:PATH"

model_names=("NAME")
save_model_names=("NAME")



export OPENAI_API_KEY=KEYS

python src/code_fix.py \
    --critic_url url \
    --critic_model_name MODEL_NAME \
    --critic_port 8008 \
    --critic_prompt prompts/CodeGemma.yaml \
    --critic_prompt_key critic_ours \
    --editor_url url\
    --editor_port 8000 \
    --editor_model_name google/codegemma-7b-it \
    --input_path data/humaneval_fix.json \
    --editor_prompt prompts/CodeGemma.yaml \
    --editor_prompt_key editor \
    --save_dir DIR \
    --feedback_gen True \
    --execution_feedback False \
    --is_iter True \
    --iter_no 2 \
    --temperature 0.1 \
    --top_p 0.95

