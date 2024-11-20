# Inductive Reasoning Elicitation for Temporal Relation Understanding

## Overview

### Paper description and main idea:
Accurately understanding temporal relations between events is a critical building block of
diverse tasks, such as temporal reading comprehension (TRC) and relation extraction (TRE).
For example in TRC, we need to understand the temporal semantic differences between the
following two questions that are lexically nearidentical:
“What finished right before the decision?” or “What finished right after the decision?”.
To discern the two questions, existing solutions have relied on answer overlaps as
a proxy label to contrast similar and dissimilar questions.
However, we claim that answer overlap can lead to unreliable results, due to spurious overlaps of two dissimilar questions with coincidentally identical answers.
To address the issue, we propose a novel approach that elicits proper reasoning behaviors through a module for predicting time spans of events.
We introduce the Timeline Reasoning Network (TRN) operating in a two-step inductive reasoning process: In the first step model initially answers each question with semantic and syntactic information.
The next step chains multiple questions on the same event to predict a timeline, which is then used to ground the answers.
Results on the TORQUE and TB-dense, TRC and TRE tasks respectively, demonstrate that TRN outperforms previous methods by effectively resolving the spurious overlaps using the predicted timeline.

### Contribution:
* We point out the spurious overlap issue in temporal relations, which arises from pointwise
timeline grounding.
* We propose the inductive solution that chains evidence for the timeline in a span-based approach.
* Our novel framework, TRN, outperforms other approaches by effectively capturing temporal relations of events.

## How to run
```
cd code
bash ./scripts/3_run_gcn_2_delib_rescon.sh
```

## Reference
```
@inproceedings{kim-etal-2024-chaining,
    title = "Chaining Event Spans for Temporal Relation Grounding",
    author = "Kim, Jongho  and
      Lee, Dohyeon  and
      Kim, Minsoo  and
      Hwang, Seung-won",
    editor = "Graham, Yvette  and
      Purver, Matthew",
    booktitle = "Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = mar,
    year = "2024",
    address = "St. Julian{'}s, Malta",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.eacl-long.101",
    pages = "1689--1700",
}
```

## Acknowledgments
This work was supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) [NO.2021-0-01343, Artificial Intelligence Graduate School Program (Seoul National University)] and Institute of Information & Communications Technology Planning & Evaluation (IITP) grant funded by the Korean government (MSIT) (No. 2022-0-00077, AI Technology Development for Commonsense Extraction, Reasoning, and Inference from Heterogeneous Data).
