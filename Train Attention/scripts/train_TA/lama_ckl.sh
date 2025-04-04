CUDA_VISIBLE_DEVICES=1 MASTER_PORT=12355 python TAALM_train_run.py \
--meta_type="lamackl" \
--val_size=16 \
--eval_batch_size=4 \
--eval_grad_accum_step=2 \
--train_data="LAMA_ckl/train_attention_traindata" \
--val_known_masking=true \
--val_interval=10 \
--batch_size=16 \
--phi_regul=true \
--phi_adapter_file="Llama-2-1b" \
--phi_model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
--theta_adapter_file="Llama-2-7b" \
--theta_model_name="meta-llama/Llama-2-7b-hf" \
--dummy_loop=1 \
--val_dummy_loop=3 \
--history_path="./checkpoints/lamackl/test_v1" \
--obj_type="phi_loss" \
--task_type="schematic" \
--TD_label="true" \
--token_max_length=512