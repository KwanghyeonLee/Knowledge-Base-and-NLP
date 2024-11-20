#!/bin/bash
export CUDA_VISIBLE_DEVICES=6

task="no_transfer"
batch=(4)
mlp_hid_size=64
seeds=(5 7 23 24 32)
model="roberta-large"
ga=1
data="tbd"
#data="matres"
checkpoint=3
lrs=(5e-6)

root="output"
export PYTHONPATH=`pwd`

for l in "${lrs[@]}"
do
  for s in "${batch[@]}"
  do
    epochs=( 10 )
    for e in "${epochs[@]}"
    do
	    for seed in "${seeds[@]}"
	    do
		    python ./te/run_singletask_te.py \
		    --task_name "${task}" \
		    --do_train \
		    --do_eval \
		    --te_type ${data} \
		    --mlp_hid_size ${mlp_hid_size} \
		    --model ${model} \
		    --data_dir ../data/ \
		    --max_seq_length 220 \
		    --train_batch_size ${s} \
		    --learning_rate ${l} \
		    --num_train_epochs ${e}  \
		    --gradient_accumulation_steps=${ga}  \
		    --output_dir ${root}/${task}_${data}/${model}_batch_${s}_lr_${l}_epochs_${e}_seed_${seed} \
		    --seed ${seed} \
            --dropout_prob 0.2 \
            --contrastive_loss_ratio 0 \
            --gcn_steps 3 \
            --graph_construction 2 \
            --use_gcn \
            --deliberate 3 \
            --deliberate_ffn 2048 \
            --residual_connection
	    done
	  done
  done
done
