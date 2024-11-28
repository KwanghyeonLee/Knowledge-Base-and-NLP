
# Large Language Models Are Clinical Reasoners: Reasoning-Aware Diagnosis Framework with Prompt-Generated Rationales

<p align="center">
  <img src="./Framework.png" width="100%" height="80%">
</p>

Official Repository of "Large Language Models Are Clinical Reasoners: Reasoning-Aware Diagnosis Framework with Prompt-Generated Rationales" accepted at AAAI 2024.

**Taeyoon Kwon\*, Kai Tzu-iunn Ong\*, Dongjin Kang, Seungjun Moon, Jeong Ryong Lee, Dosik Hwang, Beomseok Sohn, Yongsik Sim, Dongha Lee, Jinyoung Yeo**<br><sup> * Equal contribution </sup>

Paper Link: https://arxiv.org/abs/2312.07399

# Data (Privacy and Ethical Issues)
Due to privacy and ethical issues, we cannot share the data used in this work. 
However, we provide the prompts with anonymized patient information for better clarification.

## Requirements
- We used 8 NVIDIA A6000 GPUs (VRAM 48 GB).
- Make sure PyTorch (>= 1.8.0) installed. 
- Detailed requirements are listed in `conda-environment.yaml` file.

## Preprocessing
- The preprocessing code of 'ADNI' and 'AIBL' dataset are provided in `src/preprocess` directory.

## Rationalization
- For the collecting demonstration candidates we used `prompt/generate_demonstration_candidate.yaml` as our prompt.
- After collecting demonstration candidates, we used `prompt/rationalization_prompt.yaml` as our prompt to generate rationales.
- Start rationale annotation by executing `sh scripts/LLM_inference.sh`. Note that you have to set the environment variable `OPENAI_API_KEY` as your openai api key.

## Training process
- The hyperparameters are set in `/src/frozen/config/train.yaml`.
- Start the training process by executing `sh src/frozen/run_train.sh`.

## Language Model and Multi-model Model Inference
- The inference code for Multi-modal model is implemented via Pytorch Lightning.
- Start the multi-modal model inference by executing `sh scripts/frozen_inference.sh`.
- Start the language model inference by executing `sh scripts/language_model_inference.sh`.

## Acknowledgement
These works were supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT). Also, these works were supported by supported by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT).

## Citation
If you find this useful, please consider citing our paper:
```
@inproceedings{kwon2024large,
  title={Large language models are clinical reasoners: Reasoning-aware diagnosis framework with prompt-generated rationales},
  author={Kwon, Taeyoon and Ong, Kai Tzu-iunn and Kang, Dongjin and Moon, Seungjun and Lee, Jeong Ryong and Hwang, Dosik and Sohn, Beomseok and Sim, Yongsik and Lee, Dongha and Yeo, Jinyoung},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={16},
  pages={18417--18425},
  year={2024}
}
```  
