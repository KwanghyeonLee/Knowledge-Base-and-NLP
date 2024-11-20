# ContrastiveMix: Overcoming Code-Mixing Dilemma in Cross-Lingual Transfer for Information Retrieval
This repository is the implementation of our NAACL 2024 Paper [ContrastiveMix: Overcoming Code-Mixing Dilemma in Cross-Lingual Transfer for Information Retrieval](https://aclanthology.org/2024.naacl-short.17/). This work is based on [Tevatron](https://github.com/texttron/tevatron), [Pyserini](https://github.com/castorini/pyserini), dictionaries downloaded from [MUSE](https://github.com/facebookresearch/MUSE) are included in `utils/dict`.

## Overview
Multilingual pretrained language models (mPLMs) have been widely adopted in cross lingual transfer, and code-mixing has demonstrated effectiveness across various tasks in the absence of target language data. Our contribution involves an in-depth investigation into the counterproductive nature of training mPLMs on code-mixed data for information retrieval (IR). Our finding is that while code-mixing demonstrates a positive effect in aligning representations across languages, it hampers the IR-specific objective of matching representa tions between queries and relevant passages. To balance between positive and negative effects, we introduce ContrastiveMix, which disentangles contrastive loss between these conflicting objectives, thereby enhancing zero-shot IR performance. Specifically, we leverage both English and code-mixed data and employ two contrastive loss functions, by adding an additional contrastive loss that aligns embeddings of English data with their code-mixed counterparts in the query encoder. Our proposed ContrastiveMix exhibits statistically significant outperformance compared to mDPR, particularly in scenarios involving lower linguistic similarity, where the conflict between goals is more pronounced.

## Requirements
We conducted all experiments on v3-8 TPU VM with Python 3.9.12 and the following dependencies.
```
pip install torch==1.10.1 faiss-cpu==1.7.2 transformers==4.15.0 datasets==1.17.0 pyserini===0.21.0 optax==0.1.5 flax==0.6.11 chex==0.1.8 "jax[tpu]==0.4.7" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

## Training
### mDPR
```
output_dir=/path/to/output/dir

python jax_train.py \
  --do_train --output_dir ${output_dir} \
  --dataset_name Tevatron/wikipedia-nq \
  --model_name_or_path bert-base-multilingual-cased \
  --per_device_train_batch_size 16 \
  --learning_rate 1e-5 --num_train_epochs 40
```

### NaiveMix
```
lang_abbr=ar  # one of {'ar', 'bn', 'fi', 'id', 'ja', 'ko', 'ru', 'th'}
output_dir=/path/to/output/dir

python jax_train.py \
  --do_train --output_dir ${output_dir} \
  --dataset_name Tevatron/wikipedia-nq \
  --model_name_or_path bert-base-multilingual-cased \
  --per_device_train_batch_size 16 \
  --learning_rate 1e-5 --num_train_epochs 40 \
  --codemix_set en-${lang_abbr} \
  --codemix_sentence_ratio 0.2 --codemix_ratio 0.5 
```

### ContrastiveMix
```
lang_abbr=ar
output_dir=/path/to/output/dir

python jax_train.py \
  --do_train --output_dir ${output_dir} \
  --dataset_name Tevatron/wikipedia-nq \
  --model_name_or_path bert-base-multilingual-cased \
  --per_device_train_batch_size 16 \
  --learning_rate 1e-5 --num_train_epochs 40 \
  --codemix_set en-${lang_abbr} \
  --codemix_sentence_ratio_query 1 --codemix_ratio_query 0.5 \
  --contrastive --cm_loss_weight 0.1
```

## Evaluation
```
lang=arabic  
# one of {'arabic', 'bengali', 'finnish', 'indonesian', 'japanese', 'korean', 'russian', 'thai'} for Mr.TyDi
# one of {'chinese', 'hindi', 'persian', 'spanish', 'french'} for MIRACL
lang_abbr=ar
# one of {'ar', 'bn', 'fi', 'id', 'ja', 'ko', 'ru', 'th'} for Mr.Tydi
# one of {'zh', 'hi' 'fa', 'es', 'fr} for MIRACL
set_name=test  # one of {'train', 'dev', 'test'}
bm25_runfile=${output_dir}/run.bm25.mrtydi-v1.1-${lang}.${set_name}.txt
dense_runfile=${output_dir}/run.dense.mrtydi-v1.1-${lang}.${set_name}.txt
query_dataset=castorini/mr-tydi:${lang}:${set_name} # or miracl/miracl:${lang_abbr}:${set_name}
corpus_dataset=castorini/mr-tydi-corpus:${lang} # or miracl/miracl-corpus:${lang_abbr}

# encode documents
python jax_encode.py \
  --output_dir=temp \
  --model_name_or_path ${output_dir} \
  --per_device_eval_batch_size 156 \
  --dataset_name ${corpus_dataset} \
  --encoded_save_path ${output_dir}/corpus_emb_mdpr_nq.pkl

# encode queries
python jax_encode.py \
  --output_dir=temp \
  --model_name_or_path ${output_dir} \
  --per_device_eval_batch_size 1 \
  --dataset_proc_num 4 \
  --dataset_name ${query_dataset} \
  --encoded_save_path ${output_dir}/query_mdpr_nq.pkl \
  --encode_is_qry

# dense retrieve
python faiss_retriever \
  --query_reps ${output_dir}/query_mdpr_nq.pkl \
  --passage_reps ${output_dir}/corpus_emb_mdpr_nq.pkl \
  --depth 1000 \
  --batch_size -1 \
  --save_text --for_pyserini \
  --save_ranking_to ${dense_runfile}

# sparse retrieve
python -m pyserini.search --bm25 \
  --language ${lang_abbr} \
  --topics mrtydi-v1.1-${lang}-${set_name} \
  --index mrtydi-v1.1-${lang} \
  --output ${bm25_runfile} \
  --k1 0.9 \
  --b 0.4

# sparse-dense hybrid evaluation
python utils/evaluate_hybrid.py \
  --lang ${lang} --lang_abbr ${lang_abbr} \
  --sparse ${bm25_runfile} --dense ${dense_runfile} --set_name ${set_name} \
  --weight-on-dense --normalization
```

## Citation
If you find this work useful, please consider citing:
```
@inproceedings{do2024contrastivemix,
  title={ContrastiveMix: Overcoming Code-Mixing Dilemma in Cross-Lingual Transfer for Information Retrieval},
  author={Junggeun Do and Jaeseong Lee and Seung-won Hwang},
  booktitle={2024 Annual Conference of the North American Chapter of the Association for Computational Linguistics},
  year={2024},
}
```

## Acknowledgement
This research was partially supported by the MSIT (Ministry of Science and ICT), Korea, under the ITRC (Information Technology Research Center) support program (IITP-2024-2020-0-01789) supervised by the IITP (Institute for Information & Communications Technology Planning & Evaluation).
This work was also partially supported by IITP grant funded by MSIT (No.2022-0-00077, AI Technology Development for Commonsense Extraction, Reasoning, and Inference from Heterogeneous Data).
We would like to thank Google’s TPU Research Cloud (TRC) program for providing access to Cloud TPUs, and the anonymous reviewers for their reviews.
