## Can Large Language Models be Good Emotional Supporter? Mitigating Preference Bias on Emotional Support Conversation
<b>🏆 Our paper won the [Outstanding Paper Award](https://2024.aclweb.org/program/best_papers/) at ACL 2024!</b>

Official Repository of "Can Large Language Models be Good Emotional Supporter? Mitigating Preference Bias on Emotional Support Conversation" accepted at ACL 2024.

**Dongjin Kang\*, Sunghwan Kim\*, Taeyoon Kwon, Seungjun Moon, Hyunsouk Cho, Youngjae Yu, Dongha Lee, Jinyoung Yeo**<br><sup> * Equal contribution </sup>

Paper Link: https://aclanthology.org/2024.acl-long.813


## Evaluation 
To evaluate the proficiency $Q$ and preference bias $B$, run
```
python metric.py \
    --file_name /PATH/TO/RESULT/FILE
```
Check the ```result.json``` file if you need an example of the output format.

## Acknowledgement
These works were supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT). Also, these works were supported by supported by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT).

## Citation
If you find this useful, please consider citing our paper:
```
@inproceedings{kang-etal-2024-large,
    title={Can Large Language Models be Good Emotional Supporter? Mitigating Preference Bias on Emotional Support Conversation},
    author={Dongjin Kang and Sunghwan Kim and Taeyoon Kwon and Seungjun Moon and Hyunsouk Cho and Youngjae Yu and Dongha Lee and Jinyoung Yeo},
    booktitle={Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
    year={2024},
    url={https://aclanthology.org/2024.acl-long.813},
    pages={15232--15261},
}
```
