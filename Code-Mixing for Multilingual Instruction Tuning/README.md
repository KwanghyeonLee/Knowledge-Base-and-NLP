# [NAACL 2024 Findings] COMMIT: Code-Mixing English-Centric Large Language Model for Multilingual Instruction Tuning

## Overview
Recently, instruction-tuned large language models (LLMs) are showing prominent performance on various tasks, such as question answering. However, the majority of instruction-tuned LLMs are English-centric, which hinders their application to low-resource language QA. In this paper, we propose COde-Mixed Multilingual Instruction Tuning (COMMIT) to adapt English-centric LLM to low-resource language QA. We point out two main causes of English-centricness: imbalance of unlabeled data, and English-centric instruction tuning datasets. To deviate from English-centric instruction tuning, we propose to specialize code-mixing for instruction tuning, which blocks code-mixing in English templates, to leverage the potential of its superiority. To overcome data imbalance, we perform cross-lingual alignment. The majority of cross-lingual alignment works focused on making representations similar, which is not desirable to decoder-based LLMs, such as LLaMA. Therefore, we propose code-mixed continual causal language modeling to align the decoder. COMMIT improves the exact match score of low-resourced language QA by up to 32x.

## Specialized Code-mixing for Instruction Tuning 
First, prepare the weight of LLaMA by following [EasyLM doc](docs/llama.md) to, and download the [MUSE](https://github.com/facebookresearch/MUSE) dictionary to `dicts`.

Then perform COMMIT on TPUv3-8 and convert to hf format as follows:
```bash
export save_path=SAVE_PATH
export lang=hi # or el, th
export llama_weight_path=LLAMA_WEIGHT_PATH

python -m EasyLM.models.llama.llama_train \
--mesh_dim=1,1,-1 \
--load_llama_config=7b \
--load_checkpoint=params::$llama_weight_path \
--total_steps=1210 \
--logger.output_dir=$save_path \
--save_milestone_freq=1210 \
--tokenizer.vocab_file=tokenizer.model \
--tokenizer.add_bos_token=True \
--tokenizer.add_eos_token=True \
--optimizer.type=adamw \
--optimizer.accumulate_gradient_steps=1 \
--optimizer.adamw_optimizer.weight_decay=0.0 \
--optimizer.adamw_optimizer.lr=2e-5 \
--optimizer.adamw_optimizer.b1=0.9 \
--optimizer.adamw_optimizer.b2=0.999 \
--optimizer.adamw_optimizer.end_lr=1e-7 \
--optimizer.adamw_optimizer.lr_warmup_steps=36 \
--optimizer.adamw_optimizer.lr_decay_steps=1210 \
--optimizer.adamw_optimizer.bf16_momentum=True \
--train_dataset.text_processor.alpaca=True \
--train_dataset.text_processor.codemix_dict_path=dicts/en-${lang}.txt \
--train_dataset.text_processor.codemix_ratio=0.9 \
--train_dataset.text_processor.block_codemix_in_template=True \
--train_dataset.type=huggingface \
--train_dataset.huggingface_dataset.path=tatsu-lab/alpaca \
--train_dataset.huggingface_dataset.seq_length=512 \
--train_dataset.huggingface_dataset.batch_size=16 \
--update_llama_config='{"resid_pdrop": 0.05, "embd_pdrop": 0.05, "attn_pdrop": 0.05, "fcm_max_ratio": 0.1}' --log_freq=5000

export hf_save_dir=HF_SAVE_DIR
python -m EasyLM.models.llama.convert_easylm_to_hf \
--load_checkpoint=params::$save_path/*/streaming_params_1210 \
--tokenizer_path=tokenizer.model \
--model_size='7b' \
--output_dir=$hf_save_dir
```
Notes
- To build naive Alpaca, remove all the settings related to codemix.
- To run naive code-mix (Table 4), set `--train_dataset.text_processor.block_codemix_in_template=False`.

Finally, run evaluation on GPU
```bash
python lm-evaluation-harness/main.py \
--model hf-causal-experimental \
--model_args pretrained=$hf_save_dir --batch_size=2 \
--tasks ogx_xquad_${lang} --output_path $hf_save_dir/result.json \
--device cuda
```
Notes
- To run evaluation with English prompt, run with `--tasks ogx_xquad_en${lang}`. 
- You may run `ogx_mlqa` instead of `ogx_xquad`.

## Aligning before COMMIT
To align before COMMIT, run the script below before performing COMMIT:
```bash
export aligned_output_dir=ALIGNED_OUTPUT_DIR
python -m EasyLM.models.llama.llama_train \
--mesh_dim=1,1,-1 \
--load_llama_config=7b \
--load_checkpoint=params::$llama_weight_path \
--total_steps=10000 \
--logger.output_dir=$aligned_output_dir \
--save_milestone_freq=10000 \
--tokenizer.vocab_file=tokenizer.model \
--tokenizer.add_bos_token=True \
--tokenizer.add_eos_token=True \
--optimizer.type=adamw \
--optimizer.accumulate_gradient_steps=1 \
--optimizer.adamw_optimizer.weight_decay=0.0 \
--optimizer.adamw_optimizer.lr=2e-5 \
--optimizer.adamw_optimizer.b1=0.9 \
--optimizer.adamw_optimizer.b2=0.999 \
--optimizer.adamw_optimizer.end_lr=1e-7 \
--optimizer.adamw_optimizer.lr_warmup_steps=300 \
--optimizer.adamw_optimizer.lr_decay_steps=10000 \
--optimizer.adamw_optimizer.bf16_momentum=True \
--train_dataset.text_processor.fields='text' \
--train_dataset.text_processor.codemix_dict_path=dicts/${lang}-en.txt \
--train_dataset.text_processor.codemix_ratio=0.5 \
--train_dataset.type='huggingface' \
--train_dataset.huggingface_dataset.path='graelo/wikipedia' \
--train_dataset.huggingface_dataset.name="20230601.${lang}" \
--train_dataset.huggingface_dataset.seq_length=512 \
--train_dataset.huggingface_dataset.batch_size=16 \
--update_llama_config='{"resid_pdrop": 0.05, "embd_pdrop": 0.05, "attn_pdrop": 0.05, "fcm_max_ratio": 0.1}' --log_freq=5000
```
Then start with the saved checkpoint in `ALIGNED_OUTPUT_DIR` to perform COMMIT.

Notes
- To run naive CLM (Table 3), just set `--train_dataset.text_processor.codemix_ratio` as 0.

## Acknowledgement
Implemented based on [EasyLM](https://github.com/young-geng/EasyLM) and [lm-evaluation-harness](https://github.com/OpenGPTX/lm-evaluation-harness).

This research was partially supported by the MSIT (Ministry of Science and ICT), Korea, under the ITRC (Information Technology Research Center) support program (IITP-2024-2020-0-01789) supervised by the IITP (Institute for Information & Communications Technology Planning & Evaluation).
This work was also partially supported by IITP grant funded by MSIT [No.2022-0-00077, AI Technology Development for Commonsense Extraction, Reasoning, and Inference from Heterogeneous Data and No.2021-0-01343-004, Artificial Intelligence Graduate School Program (Seoul National
University)].
We would also like to thank Google’s TPU Research Cloud (TRC) program for providing Cloud TPUs.

## Reference
```
@inproceedings{lee-etal-2024-commit,
    title = "{COMMIT}: Code-Mixing {E}nglish-Centric Large Language Model for Multilingual Instruction Tuning",
    author = "Lee, Jaeseong  and Jung, YeonJoon and Hwang, Seung-won",
    editor = "Duh, Kevin  and
      Gomez, Helena  and
      Bethard, Steven",
    booktitle = "Findings of the Association for Computational Linguistics: NAACL 2024",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-naacl.198",
    doi = "10.18653/v1/2024.findings-naacl.198",
    pages = "3130--3137",
}
```