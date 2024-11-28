# Base DL (ARN detection)
python -m main --task 'test' --model_name STL --class_type 1 --device_cpu 1 --do_test

# Base DL (CMV retinitis detection) 
python -m main --task 'test' --model_name STL --class_type 2 --device_cpu 1 --do_test

# FSMTL
python -m main --task 'test' --model_name FSMTL --device_cpu 1 --do_test

# SPMTL
python -m main --task 'test' --model_name SPMTL --device_cpu 1 --do_test

# ADMTL
python -m main --task 'test' --model_name ADMTL --device_cpu 1 --do_test