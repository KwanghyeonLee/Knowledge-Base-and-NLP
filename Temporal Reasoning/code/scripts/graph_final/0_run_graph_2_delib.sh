#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

task="torque"
bs=6
epoch=10
mlp_hid_size=64
model_type="roberta-large"
ga=1
ratio=1.0
prefix="numnetfinal"
suffix="_end2end_final.json"
gcnstepss=(3)
graph_construction=2
export PYTHONPATH=`pwd`

learningrates=(1e-5)
drs=(0.2)
seeds=(7 42 123)

for l in "${learningrates[@]}"
do
    for gcnsteps in "${gcnstepss[@]}"
        do
        for seed in "${seeds[@]}"
        do
            for dropout in "${drs[@]}"
            do
            python graph/numnet/run_end_to_end_graph.py \
            --task_name "${task}" \
            --do_train \
            --do_eval \
            --mlp_hid_size ${mlp_hid_size} \
            --model_type ${model_type} \
            --data_dir ../data/ \
            --file_suffix ${suffix} \
            --train_ratio ${ratio} \
            --max_seq_length 178 \
            --train_batch_size ${bs} \
            --learning_rate ${l} \
            --seed ${seed} \
            --num_train_epochs ${epoch}  \
            --gradient_accumulation_steps ${ga}  \
            --dropout_prob ${dropout} \
            --contrastive_loss_ratio 0 \
            --gcn_steps ${gcnsteps} \
            --graph_construction ${graph_construction} \
            --use_gcn \
            --deliberate \
            --residual_connection \
            --output_dir output/${model_type}/${prefix}/batch_${bs}_lr_${l}_dr_${dropout}_epochs_${epoch}_${ratio}/seed_${seed}_gc${gcnsteps}_${graph_construction}
        done
        done
    done
done
