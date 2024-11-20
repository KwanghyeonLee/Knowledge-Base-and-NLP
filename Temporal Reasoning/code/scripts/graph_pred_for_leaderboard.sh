#!/bin/bash
tasks=("leaderboard")
epoch=10
batch=6
mlp_hid_size=64
seed=(7 42 123)
lr=(1e-5)
dr=0.2
gcnsteps=3
graph_construction=2

suffix="_end2end_final_unlabeled.json"
export PYTHONPATH=`pwd`
export CUDA_VISIBLE_DEVICES=1


model="roberta-large"

for task in "${tasks[@]}"
  do
  for l in "${lr[@]}"
    do
	  for s in "${seed[@]}"
	    do
        dir=
	    python graph/numnet/output_pred_graph.py \
	    --task_name ${task} \
	    --split "test" \
	    --model ${model} \
	    --mlp_hid_size ${mlp_hid_size} \
	    --file_suffix ${suffix} \
	    --data_dir ../data/ \
	    --model_dir output/${dir}/  \
	    --max_seq_length 178 \
	    --eval_batch_size 12 \
        --gcn_steps ${gcnsteps} \
        --dropout_prob ${dr} \
        --use_gcn \
        --graph_construction ${graph_construction} \
        --deliberate \
        --residual_connection \
	    --seed ${s}
	    done
    done
done
