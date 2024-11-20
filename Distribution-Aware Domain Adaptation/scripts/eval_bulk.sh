set -e

datasets=(
"nfcorpus"
"scifact"
"trec-covid"
"fiqa"
"robust04"
"scidocs"
)

ckpt_paths=(
"/workspace/research/logs/train/msmarco/runs/2023-10-07_09-41-40/checkpoints/step=30000-loss=3.23.ckpt"
"/workspace/research/logs/train/msmarco/runs/2023-08-31_12-54-57/checkpoints/step=70000-loss=4.16.ckpt"
#"/workspace/research/logs/train/msmarco/runs/2023-09-02_05-36-18/checkpoints/step=70000-loss=5.19.ckpt"
#"/workspace/research/logs/train/msmarco/runs/2023-09-05_05-44-21/checkpoints/step=70000-loss=6.75.ckpt"
#"/workspace/research/logs/train/msmarco/runs/2023-09-06_00-49-48/checkpoints/step=70000-loss=12.42.ckpt"
#"/workspace/research/logs/train/msmarco/runs/2023-10-05_04-59-44/checkpoints/step=70000-loss=3.03.ckpt"
)

for i in {0..5};
do
    dataset=${datasets[$i]}
    ckpt_path=${ckpt_paths[$i]}
    echo "Evaluating $dataset"
    echo "Checkpoint path: $ckpt_path"
    echo "-----------------------------------"
    python3 -m src.eval \
      experiment=eval-hybrid-curr-semantic-idf-norm \
      trainer.devices=[0,1,2,3,4,5,6,7] \
      datamodule.train_dataset_type=gpl \
      datamodule.train_dataset=$dataset \
      datamodule.test_dataset_type=beir \
      datamodule.test_datasets=[$dataset] \
      trainer.val_check_interval=1 \
      ckpt_path="'$ckpt_path'"
done
