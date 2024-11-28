# Train-Attention-Augmented Language Model (TAALM) 
---
Official Repository of "Train-Attention: Meta-Learning Where to Focus in Continual Knowledge Learning" accepted at <b>NeurIPS 2024</b>.

<b>Yeongbin Seo, Dongha Lee*, Jinyoung Yeo*</b> <sup>* indicates corresponding author.</sup>

Paper Link: https://arxiv.org/abs/2407.16920    

Overview of Train-Attention-Augmented Language Model.
<p align="center">
  <img src="./img/train-attention-nuerips2024-poster.png" width="100%" height="80%">
</p>

---
**Abstract**

Previous studies on continual knowledge learning (CKL) in Large Language Models (LLMs) have predominantly focused on approaches such as regularization, architectural modifications, and rehearsal techniques to mitigate catastrophic forgetting. However, these methods naively inherit the inefficiencies of standard training procedures, indiscriminately applying uniform weight across all tokens, which can lead to unnecessary parameter updates and increased forgetting. To address these shortcomings, we propose a novel CKL approach termed Train-Attention-Augmented Language Model (TALM), which enhances learning efficiency by dynamically predicting and applying weights to tokens based on their usefulness. This method employs a meta-learning framework that optimizes token importance predictions, facilitating targeted knowledge updates and minimizing forgetting. Through experiments conducted on both newly introduced and established CKL benchmarks, TALM proves the state-of-the-art performance upon the baselines, and also shows synergistic compatibility when integrated with the baselines. We observe that existing benchmarks do not clearly exhibit the trade-off between learning and retaining, therefore we propose a new benchmark, \textsc{LAMA-ckl}, to address this issue.

# Installation

---

```coq
$ conda create --name taalm python=3.8
$ conda activate taalm
$ pip install torch==2.1.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html
$ pip install -U -r requirements.txt
$ python -m spacy download en_core_web_sm
```

### Must Read

The **(target)** file of the conda environment must be replaced to (**replacement**).

```coq
**(target)** [taalm env dir] > site-packages > torch > nn > modules > moduel.py

**(replacement) [**project dir] > replacement > module.py
```

This replacement facilitates the injection of parameters into a model without raising an exception, thus allowing multiple differentiations during Train-Attention training. Because this replacement file is specific to torch2.1.2, if you plan to use TAALM with other versions of Torch, you should customize this replacement to match those versions.

# Quick start

---

### Training Train-Attention

```bash
# Train-Attention for LAMA_ckl
$ bash scripts/train_TA/lama_ckl.sh

# Train-Attention for TemporalWiki
$ bash scripts/train_TA/temporalwiki.sh
```

- In our study, we used single A100 (82GB) GPU
- Loss value can be observed through wandb. More detailed observations are available through  `TAALM_train_observing_history.ipynb`  file.

## Evaluation

Bash command files for evaluation with detailed configurations are in the  `scripts > eval > [bechmark name]`   directories.

**LAMA_ckl**

```bash

# TAALM
$ bash scripts/eval/lamackl/targeted.sh

# finetune
$ bash scripts/eval/lamackl/finetune.sh

# K-Adapter
$ bash scripts/eval/lamackl/kadapter.sh

# RecAdam
$ bash scripts/eval/lamackl/recadam.sh

# Mix-review
$ bash scripts/eval/lamackl/review.sh

# RHO-1
$ bash scripts/eval/lamackl/rho.sh

# Oracle
$ bash scripts/eval/lamackl/oracle.sh
```

**TemporalWiki**

```bash
# TAALM
$ bash scripts/eval/twiki/targeted.sh

...
```

- In our study, we used 8 RTX3090 (24GB) GPUs with DDP (Distributed Data Parallel)
- In the evaluation of TemporalWiki, where separate learning-evaluation phase should be conducted for total 3 periods, the experimental results are also saved in total 3 files. Each 3 files for one experiment are saved as a name format `[experiment name]_[number].pkl` (number$\in$ {0,1,2}).

# Observation on The Results

---

As the evaluation metrics in our paper are specified, we log the results in local pickle files and observe them through ipynb files, rather than utilizing wandb. We present Ipynb files for observations of each LAMA-ckl and TemporalWIki benchmark.

LAMA-ckl :  `observation_lamackl.ipynb`

TemporalWiki : `observation_twiki.ipynb`

# Pipeline for Building LAMA-ckl Dataset

---

We provide LAMA-ckl dataset which is tailored to evaluate Llama2-7B and TinyLlama-1.1B model with QLoRA and K-Adpater. But **LAMA-ckl** benchmark can be tailored for any baseline models. We present our pipeline for this. Execute the following step.

1) Download LAMA dataset from   https://github.com/facebookresearch/LAMA

```bash
wget https://dl.fbaipublicfiles.com/LAMA/data.zip
unzip data.zip
rm data.zip
```

unzip it into `/data/LAMA`  directory

2) Following all the blocks in the `LAMA_ckl_pipeline.ipynb` , convert the LAMA into the LAMA_ckl.

 On the block which is labeled ‘**Calculating Score (Accuracy)**’, you can reconfigure the code to let your desired models measure the accuracy. We present the source code. 

# Download And Prepare TemporalWiki Dataset

---

First, download TemporalWiki dataset from https://github.com/joeljang/temporalwiki/tree/main .

```bash
wget https://huggingface.co/datasets/seonghyeonye/TemporalWiki/resolve/main/Wikipedia_Full.zip

wget https://huggingface.co/datasets/seonghyeonye/TemporalWiki/resolve/main/TWiki_Diffsets.zip

wget https://huggingface.co/datasets/seonghyeonye/TemporalWiki/resolve/main/TWiki_Probes.zip
```

Manually move Wikipedia_Full forder into the path `data/Wikipedia_Full`

Filter the Diffset, referring to the Appendix C.1.   `preprocess_temporalwiki_dataset.py` file automatically filter the TemporalWiki Diffset and yield training dataset on the path  `data/TemporalWiki/train/` , and also convert the evaluation dataset (twiki_probes) into jsonl files on the `path data/TemporalWIki/eval`.

```bash
python preprocess/preprocess_temporalwiki_dataset.py
```

---
## Acknowledgement
These works were supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT). Also, these works were supported by supported by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT).
## Citation

If you find this useful, please consider citing our paper:

```
@misc{seo2024trainattentionmetalearningfocuscontinual,
      title={Train-Attention: Meta-Learning Where to Focus in Continual Knowledge Learning}, 
      author={Yeongbin Seo and Dongha Lee and Jinyoung Yeo},
      year={2024},
      eprint={2407.16920},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2407.16920}, 
}
```