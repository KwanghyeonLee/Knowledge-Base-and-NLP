set -e

CUDA_ID=$1
EXPERIMENT_NAME=$2

datasets=(
#"robust04"
"fiqa"
#"scidocs"
#"scifact"
#"nfcorpus"
)

ckpt_paths=(
#"/workspace/research/logs/train/msmarco/runs/2023-10-11_01-18-15/checkpoints/last.ckpt"
"/workspace/research/logs/train/msmarco/runs/2023-10-10_10-49-24/checkpoints/last.ckpt"
#"/workspace/research/logs/train/msmarco/runs/2023-10-10_10-44-52/checkpoints/last.ckpt"
#"/workspace/research/logs/train/msmarco/runs/2023-10-10_10-27-56/checkpoints/last.ckpt"
#"/workspace/research/logs/train/msmarco/runs/2023-10-10_10-27-45/checkpoints/last.ckpt"
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