export CUDA_VISIBLE_DEVICES=3

#!/bin/bash
task="no_transfer"
ratio=(1.0)
epoch=10
batch=(4)
mlp_hid_size=64
# seeds=(5 7 23 24 42)
seeds=(5 7 23 24 32)
learnrates=(5e-6)
model="roberta-large"
root="output"
te="tbd"
device="0"
export PYTHONPATH=`pwd`

for seed in "${seeds[@]}"
do
  for r in "${ratio[@]}"
  do
    for lr in "${learnrates[@]}"
    do
      for b in "${batch[@]}"
    do
      # code/output/no_transfer_tbd/roberta-large_batch_4_lr_5e-6_epochs_10_seed_5
      model_dir="${task}_${te}/${model}_batch_${b}_lr_${lr}_epochs_${epoch}_seed_${seed}/gcnTrue_deliberate32048_rescon"
        python ./te/eval_singletask_te.py \
        --task_name ${task} \
        --eval_ratio ${r} \
        --do_lower_case \
        --te_type ${te} \
        --model ${model} \
        --device_num ${device} \
        --mlp_hid_size ${mlp_hid_size} \
        --data_dir ../data/ \
        --model_dir ${root}/${model_dir}/\
        --max_seq_length 200 \
        --eval_batch_size 32 \
        --seed ${seed} \
        --dropout_prob 0.2 \
        --contrastive_loss_ratio 0 \
        --gcn_steps 3 \
        --graph_construction 2 \
        --use_gcn \
        --deliberate 3 \
        --deliberate_ffn 2048 \
        --residual_connection | tee ${root}/${model_dir}/eval.txt
      done
    done
  done
done