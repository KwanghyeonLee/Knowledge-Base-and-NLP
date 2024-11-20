# Script, Language, and Labels: Overcoming Three Discrepancies for Low-Resource Language Specialization [AAAI2023]

This work is implemented based on [specializing-multilingual](https://github.com/ethch18/specializing-multilingual), [bert](https://github.com/google-research/bert), [allennlp](https://github.com/ethch18/allennlp/tree/bd4457431e818cc3650e195a2b65345ee3f7c7e9)

## Overview

### Paper description and main idea:

Although multilingual pretrained models (mPLMs) enabled support of various natural language processing in diverse languages, its limited coverage of 100+ languages lets 6500+ languages remain ‘unseen’. One common approach for an unseen language is specializing the model for it as target, by performing additional masked language modeling (MLM) with the target language corpus. However, we argue that, due to the discrepancy from multilingual MLM pretraining, a naive specialization as such can be suboptimal. Specifically, we pose three discrepancies to overcome.
**Script** and **linguistic** discrepancy of the target language from the related seen languages, hinder a positive transfer, for which we propose to maximize representation similarity, unlike existing approaches maximizing overlaps. In addition, **label** space for MLM prediction can vary across languages, for which we propose to reinitialize top layers for a more effective adaptation. Experiments over four different language families and three tasks shows that our method improves the task performance of unseen languages with statistical significance, while previous approach fails to.

![Overview Image](resources/overview.png)

### Contribution:

* We provide a different metric for previously proposed script discrepancy, and propose a novel concept of 'cross-script alignment' to mitigate it.
* We reveal two more discrepancies (linguistic and label) and provide simple yet effective solutions for them.
* Our method significantly outperforms baselines, in four diverse languages and three different tasks, while the previous approach to mitigate discrepancies fails to.

## Code Running

### SL2 main exps (Table 1)
#### environments
TPU VM for bert pretraining
```bash
virtualenv cpuonly
pip install -r requirements.txt
source ~/cpuonly/bin/activate

pip install tensorflow-cpu==2.6.0 torch==1.9.1+cpu torchvision==0.10.1+cpu torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt

deactivate
```

NGC 21.03 docker for SL2 & fine-tuning
```bash
pip install nvidia-pyindex
pip install nvidia-tensorflow==1.15.5+nv21.03
pip install -r requirements
cd allennlp
pip install -e .

chmod +x hyak-allennlp-train-v2-base
chmod +x common-allennlp-eval-v2
```

#### download mBERT
```bash
curl -O https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip
unzip multi_cased_L-12_H-768_A-12.zip
mv multi_cased_L-12_H-768_A-12 bert/mbert
for file in pytorch_model.bin tokenizer_config.json config.json; do
  curl -o bert/mbert/$file https://huggingface.co/bert-base-multilingual-cased/resolve/main/$file
done
```
Then download [pytorch_model.bin](https://huggingface.co/bert-base-multilingual-cased/tree/main), place in `bert/mbert`

#### specializing tokenizer
Set `target_lang`, `translit_lang` based on the name in `specializing-multilingual-data`. Set desired `count`, which is the number of tokens to add
```bash
export target_lang=ug
export translit_lang=${target_lang}latinnfkc
export count=5000
```
Then build vocabulary augmented bert (used for VA or SL2 in Table 1),
```bash
export base_path=specializing-multilingual-data/data/${target_lang}/unlabeled
export output_dir=${base_path}/bert_cleaned
export tok_output_dir=${base_path}/bert_shards

# VA orig lang and check UNK
export exp_base_dir=specializing-multilingual-data/data/${target_lang}/unlabeled
export exp_output_dir=${exp_base_dir}/bert_cleaned
export exp_tok_output_dir=${exp_base_dir}/bert_shards
export train_corpus=${exp_base_dir}/bert_cleaned/train.txt
export base_setting="--corpus $train_corpus --base-vocab bert/scripts/convert_to_hf/mbert_vocab.txt "
for vocab_size in 5000; do
  python scripts/vocabulary/train_wordpiece_vocab.py --vocab-size $vocab_size --corpus $train_corpus --output-dir ${exp_tok_output_dir}
  python scripts/vocabulary/select_wordpieces_for_injection.py $base_setting --new-vocab ${exp_base_dir}/bert_shards/${vocab_size}-5-1000-vocab.txt --count ${count} --output-file $exp_output_dir/${vocab_size}-5-1000-${count}.txt
done

for file in bert_config.json config.json tokenizer_config.json; do
  cp bert/mbert/${file} $tok_output_dir/$file
  cp bert/mbert/${file} $tok_output_translit_dir/$file
done

export PYTHONPATH=`pwd`
tgt_lang=ug
translit_lang=${tgt_lang}latinnfkc
export base_path=specializing-multilingual-data/data/${tgt_lang}/unlabeled/bert_cleaned
export vocab_augmented_bert=$base_path/${tgt_lang}_5000-5-1000-${count}
python scripts/data/augment_bert.py --new_vocab=$base_path/5000-5-1000-${count}.txt --save_path=$vocab_augmented_bert
python scripts/data/convert_torch_to_tf.py --model_name $vocab_augmented_bert --cache_dir $vocab_augmented_bert --tf_cache_dir $vocab_augmented_bert
```

#### cross-script alignment + reset_head
We provided transliteration mapping `translit_dict`, which is simply orig_word - translit_word pair, with all the words in the wikipedia corpus of target language

```bash
export OUTPUT_DIR=YOUR_OUTPUT_DIR
export bert_init_base_path=YOUR_INPUT_PY_BERT # possibly the vocab augmented path
export align_tgt=$vocab_augmented_bert # or possibly YOUR_INPUT_PY_BERT
export lang=ug
export latin_suffix=latinnfc
export cache_suffix=$lang
python scripts/data/finetune_align_layers.py --lang $cache_suffix --align_tgt_model_tok_cfg=${align_tgt} --model_tok_cfg=$bert_init_base_path --output_dir=$OUTPUT_DIR  --tl_dict=translit_dict/${lang}_to_${lang}${latin_suffix}_tok.txt --orig_corpus=$train_corpus
```
If you don't want to reset head, you can set `--reset_head=False`

#### cross-ling alignment + reset_head
```bash
export OUTPUT_DIR=YOUR_OUTPUT_DIR
export bert_init_base_path=YOUR_INPUT_PY_BERT
export lang=ug
export latin_suffix=latinnfc
export cache_suffix=$lang
export corpus_base="scripts/data/clalign/tr-ug.txt/Tanzil.tr-ug"
export src_lang=tr

cp $vocab_augmented_bert/*.json $bert_init_base_path/
cp $vocab_augmented_bert/*.txt $bert_init_base_path/

export basic_set="--cross_ling --use_basic_tok "
export data_args="--orig_corpus ${corpus_base}.${lang} --src_corpus ${corpus_base}.${src_lang} --word_alignment ${corpus_base}grow-diag-final.basictok"
python scripts/data/finetune_align_layers.py $basic_set $data_args --align_tgt_model_tok_cfg=${align_tgt} --lang ${cache_suffix}  --model_tok_cfg=${bert_init_base_path} --output_dir=${OUTPUT_DIR}
```

#### prepare them for tf bert pretrain
```bash
python scripts/data/convert_torch_to_tf.py --model_name $OUTPUT_DIR --cache_dir $OUTPUT_DIR --tf_cache_dir $OUTPUT_DIR
gsutil -m cp -r $OUTPUT_DIR/\* GS_BUCKET_SAVE_PATH/
```
now `GS_BUCKET_SAVE_PATH` will contain a .ckpt (which consists of 3 files with different suffixes) . We name it as `GS_BUCKET_INIT_CKPT`

#### bert pretrain on tf
Create tfrecord
```bash
export tfrecord_on_gs_bucket=GS_BUCKET_TFRECORD_PATH
export lang=ug # or uglatinnfc
export PYTHONPATH=`pwd`

export tfrecord_file=${lang}/train.tfrecord
/root/cpuonly/bin/python bert/create_pretraining_data.py --input_file=$train_corpus --vocab_file=$vocab_augmented_bert/vocab.txt --output_file=${tfrecord_file} --do_lower_case=False --max_seq_length=128 --random_seed 13370 --dupe_factor=5 --max_predictions_per_seq=20 --masked_lm_prob=0.15 
gsutil cp ${tfrecord_file} $tfrecord_on_gs_bucket
```
copy `vocab_augmented_bert` into TPU VM also. Then, on TPU VM, run pretraining:

```bash
export tfrecord_on_gs_bucket=GS_BUCKET_TFRECORD_PATH
export init_checkpoint=GS_BUCKET_INIT_CKPT
export output_dir=GS_BUCKET_OUTPUT_DIR
export PYTHONPATH=`pwd`
export dir_settings="--init_checkpoint=$init_checkpoint --input_file=$tfrecord_on_gs_bucket --output_dir=$output_dir --bert_config_file=$vocab_augmented_bert/bert_config.json "
export train_params="--train_batch_size=16 --num_train_epoch=20 --num_warmup_steps=1000 --learning_rate=2e-5 --save_checkpoints_epoch=1  --epochs_per_eval=1 "
export train_settings="--do_train=True --use_tpu=True --tpu_name=local --iterations_per_loop=5000 "
python bert/run_pretraining_nonsp.py  --seed 42 ${dir_settings} ${train_params} ${train_settings} 

export local_save_dir=YOUR_SAVE_DIR
mkdir -p /dev/shm/$local_save_dir/
gsutil -m cp -r $output_dir/\* /dev/shm/$local_save_dir/
${HOME}/cpuonly/bin/python scripts/modeling/convert_script.py --output_dir=/dev/shm/$local_save_dir --bert_config_path=$vocab_augmented_bert --last_only
```
Then the converted model will be placed in `/dev/shm/$local_save_dir/epoch_last`. Copy that to ngc docker

#### fine-tune
Modify [PATHFINDER](config/autogen/DO_NOT_ERASE_pathfinder.libsonnet) to point right mtl.libsonnet path

Then, run followings (if you run on myv, make sure to set FINAL_FOLD and TOTAL_SIZE)
```bash
export PRETRAIN_PATH=YOUR_PT_BERT_PATH
export SUFFIX=SOME_SUFFIX_YOU_LIKE_TO_MAKE_UNIQUE_EXP_DIR
export FINAL_FOLD=4 # set 6 for myv, for 8-fold test with 0-6 train and 7 valid
export TOTAL_SIZE= # set 1403 for myv
export PYTHONPATH=`pwd`
./hyak-allennlp-train-v2-base mtlpos_ug_tva_best
./common-allennlp-eval-v2 mtlpos_ug_tva_best
```
You may change `mtlpos_ug_tva_best` as the following name in `config/autogen`

#### get p-value
```python
from scipy import stats
import numpy as np
current = np.array(ours)
others = np.array(others)
greater = stats.ttest_1samp(current - others, 0, alternative='greater').pvalue
```

### Other exps
#### alignments after specialization (Table 3)
run cross-ling exp with additional params:
`--align_tgt=SPECIALIZED_MODEL --reset_head=False --update_src_emb_also=True --reduce_word_strategy last --regularlize_on_orig_embeddings=True`

#### reinit more (Table 4)
use `scripts/modeling/reinit_layers.py`

#### analysis (representation sim)
for visualization & hausdorff distance,
```bash
pip install hausdorff plotly
```
Then use `scripts/analysis/get_hausdorff_and_datas.py` then `scripts/analysis/vis_only_some.py`

#### analysis (label discrep)
Collect the label predictions. Then use `scripts/analysis/check_subword_frequencies.py` then `scripts/analysis/check_mispred_how.py`

## Reference
If you find this code helpful, please consider citing:
```
@article{Lee_Lee_Hwang_2023,
	author = {Lee, Jaeseong and Lee, Dohyeon and Hwang, Seung-won},
	doi = {10.1609/aaai.v37i11.26528},
	journal = {Proceedings of the AAAI Conference on Artificial Intelligence},
	title = {Script, Language, and Labels: Overcoming Three Discrepancies for Low-Resource Language Specialization},
	url = {https://ojs.aaai.org/index.php/AAAI/article/view/26528},
	volume = {37},
	year = {2023}
}
```

## Acknowledgement
This research was supported by the MSIT (Ministry of Science and ICT), Korea, under the ITRC (Information
Technology Research Center) support program (IITP-2022-2020-0-01789) supervised by the IITP(Institute for Information & Communications Technology Planning & Evaluation).
This work was also supported by IITP grants funded by the MSIT [No.2022-0-00077, AI Technology Development
for Commonsense Extraction, Reasoning, and Inference from Heterogeneous Data], [NO.2021-0-0268, Artificial Intelligence Innovation Hub (Artificial Intelligence Institute, Seoul National University)], and [NO.2021-0-01343, Artificial Intelligence Graduate School Program (Seoul National University)].
We would like to thank Google’s TPU Research Cloud (TRC) program for providing Cloud TPUs.
