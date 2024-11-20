set -e
apt-get install -y sshpass

gpl_path="/workspace/gpl"

new_curr_path="curriculum-gpl-training-data.tsv"
leg_curr_path="curriculum-gpl-training-data.tsv.zero-idf.zero-prob"

datasets=(
"robust04"
"scifact"
"scidocs"
"nfcorpus"
"fiqa"
)

for i in {0..4};
do
    dataset=${datasets[$i]}
    new_pth="${gpl_path}/${dataset}/${new_curr_path}"
    leg_pth="${gpl_path}/${dataset}/${leg_curr_path}"
    echo "-----------------------------------"
    echo "Dataset: $dataset"
    echo "New Path: $new_pth"
    echo "Leg Path: $leg_pth"
    echo "-----------------------------------"
    #

    mv $new_pth $leg_pth

done