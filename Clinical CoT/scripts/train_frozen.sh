#!/bin/bash

python src/frozen/main.py \
    --config /src/frozen/config/train.yaml \
    --config_name key/from/config/file \




# WITHOUT RATIONALE ##
python src/frozen/main.py \
    --config /src/frozen/config/train_split.yaml \
    --config_name key/from/config/file \
    --train_wo_rationale \
    # --data_portion 75 for 75% of data
