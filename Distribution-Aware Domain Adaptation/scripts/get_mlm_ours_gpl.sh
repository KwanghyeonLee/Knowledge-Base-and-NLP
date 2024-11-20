set -e

CUDA_ID=$1
EXPERIMENT_NAME=$2

datasets=(
#"nfcorpus"
#"scifact"
#"scidocs"
"fiqa"
#"robust04"
)

ckpt_paths=(
#"/workspace/research/logs/train/msmarco/runs/2023-10-07_09-41-40/checkpoints/step=30000-loss=3.23.ckpt"
#"/workspace/research/logs/train/msmarco/runs/2023-08-31_12-54-57/checkpoints/step=70000-loss=4.16.ckpt"
#"/workspace/research/logs/train/msmarco/runs/2023-10-05_04-59-44/checkpoints/step=70000-loss=3.03.ckpt"
"/workspace/research/logs/train/msmarco/runs/2023-09-05_05-44-21/checkpoints/step=70000-loss=6.75.ckpt"
#"/workspace/research/logs/train/msmarco/runs/2023-09-06_00-49-48/checkpoints/step=70000-loss=12.42.ckpt"
)

for i in {0..2};
do
    dataset=${datasets[$i]}
    ckpt_path=${ckpt_paths[$i]}
    echo "-----------------------------------"
    echo "experiment name: ${EXPERIMENT_NAME}"
    echo "MLMing... $dataset"
    echo "Checkpoint path: $ckpt_path"
    echo "-----------------------------------"
    python3 -m toy.dada_ablation.amnesia.get_doc_mlm \
      experiment=eval-hybrid-curr-semantic-idf-norm \
      trainer.devices=[${CUDA_ID}] \
      datamodule.train_dataset_type=gpl \
      datamodule.train_dataset=${dataset} \
      datamodule.test_dataset_type=beir \
      datamodule.test_datasets=[${dataset}] \
      datamodule.test_batch_size=128 \
      student=${EXPERIMENT_NAME} \
      ckpt_path="'${ckpt_path}'"
done