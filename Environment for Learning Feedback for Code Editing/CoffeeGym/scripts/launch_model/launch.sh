A=0,1,2,3,4,5,6,7
B="MODEL_NAME_OR_PATH"
C=8 # length of A

CUDA_VISIBLE_DEVICES=$A
MODEL_NAME_OR_PATH=$B
TENSOR_PARALLEL_SIZE=$C

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_NAME_OR_PATH \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --seed 42 \
    --port 8000 \
    --max-model-len 4196
