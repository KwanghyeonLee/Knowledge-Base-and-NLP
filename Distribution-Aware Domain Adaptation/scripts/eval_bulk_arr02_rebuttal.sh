set -e
export WANDB_MODE=offline

methods=(
# Done #"dada"
# Done #"gpl"#"tasb" "fiqa"
# Done #"gpl"#"tasb" "scifact"
# Done #"dada"#"tasb" "fiqa"
# Done #"dada"#"tasb" "scifact"
# Done #"gpl"#"tasb" "nfcorpus"
# Done #"gpl"#"tasb" "scidocs"
#"gpl" #"tasb" "fiqa"
#Done"dada" #"tasb" "nfcorpus"
#Done"dada" #"tasb" "scifact"
"gpl" #"tasb" "scifact"
#Done "dada" #"tasb" "scidocs"
#Done "dada" #"tasb" "fiqa"
# Done "gpl"
# Done "gpl"
# Done "gpl"
# Done "gpl"
# Done "gpl"
# Done "gpl"
# Done "gpl"
# Done "gpl"
# Done "dada"
# Done "dada"
# Done "dada"
# Done "dada"
# Done "dada"
# Done "gpl"
)

model_types=(
# Done #"cond"
# Done #"tasb" "fiqa"
# Done #"tasb" "scifact"
# Done #"tasb" "fiqa"
# Done #"tasb" "scifact"
# Done #"tasb" "nfcorpus"
# Done #"tasb" "scidocs"
#"tasb" # "fiqa"
#Done"tasb" # "nfcorpus"
#Done"tasb" # "scifact"
"tasb" # "scifact"
#Done "tasb" # "scidocs"
#Done "tasb" # "fiqa"
# Done "cond"
# Done "cond"
# Done "cond"
# Done "coco"
# Done "coco"
# Done "coco"
# Done "coco"
# Done "coco"
# Done "coco"
# Done "coco"
# Done "coco"
# Done "coco"
# Done "coco"
# Done "coco"
)

datasets=(
# Done #"scidocs"
# Done #"fiqa"#"tasb" "fiqa"
# Done #"scifact"#"tasb" "scifact"
# Done #"fiqa"#"tasb" "fiqa"
# Done #"scifact"#"tasb" "scifact"
# Done #"nfcorpus"#"tasb" "nfcorpus"
# Done #"scidocs"#"tasb" "scidocs"
#"fiqa" #"tasb" "fiqa"
#Done"nfcorpus" #"tasb" "nfcorpus"
#Done"scifact" #"tasb" "scifact"
"scifact" #"tasb" "scifact"
#Done "scidocs" #"tasb" "scidocs"
#Done "fiqa" #"tasb" "fiqa"
# Done "scifact"
# Done "fiqa"
# Done "nfcorpus"
# Done "robust04"
# Done "scidocs"
# Done "fiqa"
# Done "nfcorpus"
# Done "scifact"
# Done "scidocs"
# Done "robust04"
# Done "fiqa"
# Done "nfcorpus"
# Done "scifact"
# Done "scifact"
)

ckpt_paths=(
#Done #"/workspace/research/logs/train/msmarco/runs/2024-03-27_23-53-08/checkpoints"
#Done #"/workspace/research/logs/train/msmarco/runs/2024-02-16_11-25-30/checkpoints" #"fiqa"#"tasb" "fiqa"
#Done #"/workspace/research/logs/train/msmarco/runs/2024-02-16_11-25-29/checkpoints" #"scifact"#"tasb" "scifact"
#Done #"/workspace/research/logs/train/msmarco/runs/2024-02-15_19-30-08/checkpoints" #"fiqa"#"tasb" "fiqa"
#Done #"/workspace/research/logs/train/msmarco/runs/2024-02-15_19-29-39/checkpoints" #"scifact"#"tasb" "scifact"
#Done #"/workspace/research/logs/train/msmarco/runs/2024-02-14_20-27-09/checkpoints" #"nfcorpus"#"tasb" "nfcorpus"
#Done #"/workspace/research/logs/train/msmarco/runs/2024-02-14_20-25-11/checkpoints" #"scidocs"#"tasb" "scidocs"
#"/workspace/research/logs/train/msmarco/runs/2024-02-14_16-15-00/checkpoints" #"fiqa"#"tasb" "fiqa"
#Done "/workspace/research/logs/train/msmarco/runs/2024-02-14_09-15-37/checkpoints" #"nfcorpus"#"tasb" "nfcorpus"
#Done "/workspace/research/logs/train/msmarco/runs/2024-02-14_08-44-51/checkpoints" #"scifact"#"tasb" "scifact"
"/workspace/research/logs/train/msmarco/runs/2024-02-14_07-58-21/checkpoints" #"scifact"#"tasb" "scifact"
#Done "/workspace/research/logs/train/msmarco/runs/2024-02-13_22-39-36/checkpoints" #"scidocs"#"tasb" "scidocs"
#Done "/workspace/research/logs/train/msmarco/runs/2024-02-13_22-38-58/checkpoints" #"fiqa"#"tasb" "fiqa"
# Done "/workspace/research/logs/train/msmarco/runs/2024-02-13_02-05-07/checkpoints"
# Done "/workspace/research/logs/train/msmarco/runs/2024-02-13_01-37-56/checkpoints"
# Done "/workspace/research/logs/train/msmarco/runs/2024-02-12_13-19-39/checkpoints"
# Done "/workspace/research/logs/train/msmarco/runs/2024-02-11_04-33-51/checkpoints"
# Done "/workspace/research/logs/train/msmarco/runs/2024-02-11_00-32-38/checkpoints"
# Done "/workspace/research/logs/train/msmarco/runs/2024-02-10_13-18-54/checkpoints"
# Done "/workspace/research/logs/train/msmarco/runs/2024-02-10_13-17-58/checkpoints"
# Done "/workspace/research/logs/train/msmarco/runs/2024-02-09_15-13-31/checkpoints"
# Done "/workspace/research/logs/train/msmarco/runs/2024-02-08_10-55-08/checkpoints"
# Done "/workspace/research/logs/train/msmarco/runs/2024-02-07_11-24-08/checkpoints"
# Done "/workspace/research/logs/train/msmarco/runs/2024-02-05_10-09-48/checkpoints"
# Done "/workspace/research/logs/train/msmarco/runs/2024-02-04_11-48-49/checkpoints"
# Done "/workspace/research/logs/train/msmarco/runs/2024-01-28_09-08-34/checkpoints"
# Done "/workspace/research/logs/train/msmarco/runs/2024-01-27_11-19-16/checkpoints"
)

errors=()

for ((i=0; i< ${#ckpt_paths[@]}; i++ ));
do
    model_type=${model_types[i]}
    method=${methods[i]}
    dataset=${datasets[i]}
    ckpt_path="${ckpt_paths[i]}/last.ckpt"

    # check ckpt exists
    if [ ! -f $ckpt_path ]; then
        echo "Checkpoint not found: $ckpt_path"
        errors+=("$dataset $method $model_type $ckpt_path")
        continue
    fi

    if [ $method == "dada" ]; then
        eval_experiment="eval-hybrid-curr-semantic-idf-norm"
    elif [ $method == "gpl" ]; then
        eval_experiment="eval-gpl-baseline"
    else
        echo "Method not found"
        echo $method
        continue
    fi
    eval_experiment="${eval_experiment}-${model_type}"

    echo "-----------------------------------"
    echo "Evaluating $dataset"
    echo "Checkpoint path: $ckpt_path"
    echo "Method: $method"
    echo "Model: $model_type"
    echo "-----------------------------------"
    python3 -m src.eval \
      experiment=$eval_experiment \
      trainer.devices=[4,5,6,7] \
      trainer.val_check_interval=1 \
      ckpt_path="'$ckpt_path'" \
      method_name=$method \
      model_type=$model_type \
      dataset=$dataset
done

echo "Errors:"
for error in "${errors[@]}"
do
    echo $error
done


