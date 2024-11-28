variations1=("0901-1001" "1001-1101" "1101-1201")
variations2=("0910" "1011" "1112")

length=${#variations1[@]}

for ((i=0; i<length; i++)); do
    var1=${variations1[i]}
    var2=${variations2[i]}
    echo "$var1 and $var2"

    
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MASTER_PORT=12354 python evaluation_run.py \
    --meta_type="twiki" \
    --gpu_train_batch_size=4 \
    --train_grad_accum_step=2 \
    --train_lr=1e-4 \
    --max_epochs=1 \
    --train_grad_accum="true" \
    --theta_adapter_file="Llama-2-1b" \
    --theta_model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
    --phi_adapter_file="Llama-2-1b" \
    --phi_model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
    --phi_checkpoint="checkpoints/twiki/test_v1/checkpoint_1080.pkl"  \
    --eval_batch_size=8 \
    --test_type="finetune" \
    --model_type="qlora" \
    --recadam="true" \
    --train_data="data/TemporalWiki/train/diffset_${var2}_filtered.jsonl" \
    --eval_data_changed="data/TemporalWiki/eval/${var1}_changed.jsonl" \
    --eval_data_unchanged="data/TemporalWiki/eval/${var1}_unchanged.jsonl" \
    --review_data="none" \
    --add_to_title="" \
    --twiki_num=${i} \
    --twiki_temp_save="true" \
    --twiki_num_last=$((length - 1)) \
    --token_max_length=512 \
    --bf16="true" \

done