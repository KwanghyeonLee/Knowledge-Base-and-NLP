# Base DL (ARN detection) 
python -m main --task train --model_name STL --class_type 1 --lr 0.001 --epoch 20

# Base DL (CMV retinitis detection) 
python -m main --task train --model_name STL --class_type 2 --lr 0.001 --epoch 20

# FSMTL
python -m main --task train --model_name FSMTL --alpha 0.375 --lr 0.001 --epoch 30

# SPMTL
python -m main --task train --model_name SPMTL --alpha 0.05 --batch_size 128 --lr 0.001 --epoch 30

# ADMTL
python -m main --task train --model_name ADMTL --epoch 90 --alpha 0.3 --batch_size 64 --lr 0.005 --lambda_adv 0.05 --gamma 0.005