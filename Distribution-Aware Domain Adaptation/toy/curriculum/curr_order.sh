data=$1
cuda=$2
dtype=gpl
idf_name="idfs.json"

python3 -m toy.curriculum.curriculum_ordering ${data} --cuda-id ${cuda} --dtype ${dtype} --idf-name ${idf_name}
python3 -m toy.curriculum.curriculum_ordering_merge ${data} --cuda-id ${cuda} --dtype ${dtype}
