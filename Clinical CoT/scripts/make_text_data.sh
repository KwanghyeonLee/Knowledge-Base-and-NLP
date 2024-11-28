#!/bin/bash

python src/preprocess/make_text_data.py \
    --input_path input/data/dir \
    --prompt prompt/something/.yaml/file \
    --prompt_key key/from/.yaml/file \
    --save_dir save/data/dir 
    