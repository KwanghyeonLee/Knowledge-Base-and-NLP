#!/bin/bash
export CUDA_VISIBLE_DEVICES=3

task="no_transfer"
batch=(4)
mlp_hid_size=64
seeds=(5 7 23)
model="roberta-large"
ga=1
data="tbd"
checkpoint=3
lrs=(5e-6)

root="output"

for l in "${lrs[@]}"
do
  for s in "${batch[@]}"
  do
    epochs=( 10 )
    for e in "${epochs[@]}"
    do
	    for seed in "${seeds[@]}"
	    do
		    python ./code/run_singletask_te.py \
		    --task_name "${task}" \
		    --do_train \
		    --do_eval \
		    --te_type ${data} \
		    --mlp_hid_size ${mlp_hid_size} \
		    --model ${model} \
		    --data_dir ./data/ \
		    --max_seq_length 220 \
		    --train_batch_size ${s} \
		    --learning_rate ${l} \
		    --num_train_epochs ${e}  \
		    --gradient_accumulation_steps=${ga}  \
		    --output_dir ${root}/${task}_${data}_${model}_batch_${s}_lr_${l}_epochs_${e}_seed_${seed} \
		    --seed ${seed}
	    done
	  done
  done
done
