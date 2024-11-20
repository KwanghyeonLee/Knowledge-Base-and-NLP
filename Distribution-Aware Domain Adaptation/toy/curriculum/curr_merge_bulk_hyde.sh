cuda=$1
dtype=hyde
idf_name="idfs.json"

datasets=(
"scifact"
"scidocs"
"nfcorpus"
"fiqa"
)

for i in {0..4};
do
    data=${datasets[i]}
    python3 -m toy.curriculum.curriculum_ordering_merge ${data} --cuda-id ${cuda} --dtype ${dtype}

done