# [NAACL 2024] Script-mix: Mixing Scripts for Low-resource Language Parsing

## Overview
Despite the success of multilingual pretrained language models (mPLMs) for tasks such as dependency parsing (DEP) or part-of-speech (POS) tagging, their coverage of 100s of lan guages is still limited, as most of the 6500+ languages remains "unseen." To adapt mPLMs for including such unseen langs, existing work has considered transliteration and vocabulary augmentation. Meanwhile, the consideration of combining the two has been surprisingly lacking. To understand why, we identify both complementary strengths of the two, and the hurdles to realizing it. Based on this observation, we propose ScriptMix, combining two strengths, and overcoming the hurdle. Specifically, ScriptMix a) is trained with dual-script corpus to combine strengths, but b) with sep arate modules to avoid gradient conflict. In combining modules properly, we also point out the limitation of the conventional method AdapterFusion, and propose AdapterFusion+ to overcome it. ScriptMix improves the POS accuracy by up to 14%, and improves the DEP LAS score by up to 5.6%.

## LA training (at TPU VM)
```bash
pip install -r requirements.txt
pip install -r examples/pytorch/language-modeling/requirements.txt

export model_name=TOKENIZER_SPECIALIZED_MBERT
export unlabelled_data_dir=PATH_TO_UNLABELLED_DATA
export output_dir=OUTPUT_DIR
export XRT_TPU_CONFIG="localservice;0;localhost:51011"
export train_bs=8 
python examples/pytorch/xla_spawn.py --num_cores 8 examples/pytorch/language-modeling/run_mlm.py --train_file ${unlabelled_data_dir}/all.txt --max_steps 50000 --save_strategy steps --save_steps 50000 --max_seq_length 512      --model_name_or_path $model_name     --do_train     --per_device_train_batch_size $train_bs     --learning_rate 1e-4     --output_dir $output_dir     --save_total_limit 2     --pad_to_max_length        --train_adapter   --adapter_reduction_factor 2  --adapter_config "pfeiffer+inv"
```
To train Language module for UniPELT, change `--adapter_reduction_factor 2  --adapter_config "pfeiffer+inv"` into `--reduction_r 8  --adapter_config "unipelt+inv"`

## Training single LA with Dual-script
```bash
export va_data_dir=PATH_TO_VA_DATA
export tl_data_dir=PATH_TO_TL_DATA
export output_dir=OUTPUT_DIR
export XRT_TPU_CONFIG="localservice;0;localhost:51011"
export train_bs=1 # with grad_accum 2 and len_per_language 4, per_device_batch_size is 8 actually
python examples/pytorch/xla_spawn.py --num_cores 8 examples/pytorch/language-modeling/run_mlm.py --languages 0 1 --train_files ${va_data_dir}/all.txt ${tl_data_dir}/all.txt --len_per_language 4 --do_shuffle=False --max_steps 50000 --save_strategy steps --save_steps 50000 --max_seq_length 512      --model_name_or_path $model_name     --do_train     --per_device_train_batch_size $train_bs     --learning_rate 1e-4     --output_dir $output_dir    --save_total_limit 2     --pad_to_max_length     --gradient_accumulation_steps 2     --train_adapter     --adapter_reduction_factor 2     --adapter_config "pfeiffer+inv"
```

## Fine-tuning MAD-X (at NGC 21.03 docker)

### DEP
```bash
export PYTHONPATH=`pwd`
export model_name=TOKENIZER_SPECIALIZED_MBERT
export output_dir=OUTPUT_DIR
export eval_lang=EVAL_LANG
export train_bs=16
export grad_accum=2
export fp16=True
export epochs=30
export lr=5e-4

export adapter_config=pfeiffer # or unipelt
export train_dir_suffix= # or 0~6 if cross-valid with full data, or fewshot0~6 if fewshot cross-valid
export data_size_suffix= # or _16 if 16-shot, _32 if 32-shot
export data_setting="--task_name dep --max_seq_length 256 --train_file datas/${eval_lang}/ud${train_dir_suffix}/train${data_size_suffix}.json --validation_file datas/${eval_lang}/ud${train_dir_suffix}/dev${data_size_suffix}.json --test_file datas/${eval_lang}/ud${train_dir_suffix}/test.json "
export adapter_setting="--train_adapter=True --adapter_config=${adapter_config} --load_lang_adapter ${lang_adapter_dir} --language ${eval_lang}"
export train_setting="--gradient_accumulation_steps ${grad_accum} --metric_score las --learning_rate $lr --fp16=$fp16 --evaluation_strategy=epoch --save_strategy=epoch --save_total_limit=2 --metric_for_best_model=las --greater_is_better=True --num_train_epochs=$epochs --per_device_train_batch_size $train_bs --model_name_or_path $model_name --output_dir ${output_dir}"

export lang_adapter_dir=PATH_TO_TRAINED_LANG_MODULE

python3 examples/pytorch/dependency-parsing/run_udp.py --do_train=True --do_predict=True --per_device_eval_batch_size=8 $adapter_setting $train_setting $data_setting
python3 examples/pytorch/dependency-parsing/run_udp.py --do_train=False --do_predict=True --load_best --per_device_eval_batch_size=8 $adapter_setting $train_setting $data_setting
```

### POS
```bash
export PYTHONPATH=`pwd`
export model_name=TOKENIZER_SPECIALIZED_MBERT
export output_dir=OUTPUT_DIR
export eval_lang=EVAL_LANG
export train_bs=16
export grad_accum=2
export fp16=True
export epochs=30
export lr=5e-4

export adapter_config=pfeiffer # or unipelt
export train_dir_suffix= # or 0~6 if cross-valid with full data, or fewshot0~6 if fewshot cross-valid
export data_size_suffix= # or _16 if 16-shot, _32 if 32-shot
export data_setting="--task_name pos --max_seq_length 256 --train_file datas/${eval_lang}/ud${train_dir_suffix}/train${data_size_suffix}.conllu --validation_file datas/${eval_lang}/ud${train_dir_suffix}/dev${data_size_suffix}.conllu --test_file datas/${eval_lang}/ud${train_dir_suffix}/test.conllu "
export train_setting="--gradient_accumulation_steps ${grad_accum} --overwrite_cache --learning_rate $lr --fp16=$fp16 --evaluation_strategy=epoch --save_strategy=epoch --save_total_limit=2 --metric_for_best_model=accuracy --greater_is_better=True --num_train_epochs=$epochs --per_device_train_batch_size $train_bs --model_name_or_path $model_name --output_dir $output_dir "

export lang_adapter_dir=PATH_TO_TRAINED_LANG_MODULE
export adapter_setting="--train_adapter=True --adapter_config=${adapter_config} --load_lang_adapter ${lang_adapter_dir} --language ${eval_lang}"

python3 examples/pytorch/token-classification/run_pos.py --do_train=True --do_predict=True --per_device_eval_batch_size=8 $adapter_setting $train_setting $data_setting
python3 examples/pytorch/token-classification/run_pos.py --do_train=False --do_predict=True --load_best --per_device_eval_batch_size=8 $adapter_setting $train_setting $data_setting
```

## Fine-tuning with script-mix
change the last lines of the MAD-X fine-tuning script as follows:
```bash
export lang_adapter="PATH_TO_VA_TRAINED_LANG_MODULE PATH_TO_TL_TRAINED_LANG_MODULE "
export fuse_mode="--fuse_mode lang_fuse_train_fuse_and_task " # add "--mlp_fusion_mode=key" to use AdapterFusion+/UniPELTFusion+
export adapter_setting="${fuse_mode} --train_adapter=True --adapter_config=${adapter_config}  --load_lang_adapter ${lang_adapter_dir} --language VA_LANG TL_LANG "

python3 examples/pytorch/dependency-parsing/run_udp_fusion.py --do_train=True --do_predict=True --per_device_eval_batch_size=8 $task_setting $train_setting $data_setting
python3 examples/pytorch/dependency-parsing/run_udp_fusion.py --do_train=False --do_predict=True --load_best --per_device_eval_batch_size=8 $task_setting $train_setting $data_setting
```

## Acknowledgement
This work is implemented based on [adapter-transformers](https://github.com/adapter-hub/adapter-transformers).

This research was partially supported by the MSIT (Ministry of Science and ICT), Korea, under the ITRC (Information Technology Research Center) support program (IITP-2024-2020-0-01789) supervised by the IITP (Institute for Information & Communications Technology Planning & Evaluation).
This work was also partially supported by IITP grant funded by MSIT (No.2022-0-00077, AI Technology Development for Commonsense Extraction, Reasoning, and Inference from Heterogeneous Data).
We would also like to thank Google’s TPU Research Cloud (TRC) program for providing Cloud TPUs.
