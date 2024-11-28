#!/bin/bash

Convert the model first
echo "Convert the model first"
python src/frozen/convert.py \
    --input_path pytorch_lightning/model/checkpoint \
    --output_path save/checkpoint/with/.pt

# Run inference
echo "Run inference"
python src/frozen/main.py \
    --config MedicalReasoning/src/frozen/config/inference.yaml \
    --config_name config/name/from/inference.yaml/file \
    --pretrained_path save/checkpoint/with/.pt 

# Collect inference files
echo "Collect inference files"
python src/frozen/cleanup.py \
    --input_path prediction/file/from/inference.yaml/file \
    --org_path path/to/ADNI/test_data/path \
    --org_AIBL_path path/to/AIBL/test_data/path \
    --output_path save/path/for/cleanuped/inference/data/test.json/file

# Get metrics
echo "Get metrics"
python src/train_llm/get_diagnosis_acc.py \
    --input_path save/path/for/cleanuped/inference/data/test.json/file

echo "Get metrics AIBL"
python src/train_llm/get_diagnosis_acc.py \
    --input_path save/path/for/cleanuped/inference/data/test_AIBL.json/file