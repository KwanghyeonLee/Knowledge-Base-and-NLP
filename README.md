# Knowledge-Base-and-NLP

#### This repository organizes researches related to AI Technology Development for Commonsense Extraction, Reasoning, and Inference from Heterogeneous Data, especially Knowledge-Base-and-NLP task.
#### This repository summarizes following researches.

## Research list
* CoTEVer: Chain of Thought Prompting Annotation Toolkit for Explanation Verification (EACL 2023) - Seungone Kim, Se June Joo, Yul Jang, Hyungjoo Chae, and Jinyoung Yeo.

  * The proposed Chain of Thought Prompting Annotation Toolkit for Explanation Verification (CoTEVer), is a tool-kit for annotating the factual correctness of generated explanations and collecting revision data of wrong explanations.

* Mind the Gap! Injecting Commonsense Knowledge for Abstractive Dialogue Summarization (COLING 2022) - Seungone Kim, Se June Joo, Hyungjoo Chae, Chaehyeong Kim, Seung-won Hwang, and Jinyoung Yeo.

  * The proposed Summarizing with Injected Commonsense Knowledge (SICK), is a framework that uses commonsense inferences as additional context. SICK leverages the unique characteristics of dialogues sharing commonsense knowledge across participants, to resolve the difficulties in summarizing them.

* Dialogue Chain-of-Thought Distillation for Commonsense-aware Conversational Agents (EMNLP 2023) - Hyungjoo Chae, Yongho Song, Kai Tzu-iunn Ong, Taeyoon Kwon, Minjin Kim, Youngjae Yu, Dongha Lee, Dongyeop Kang, and Jinyoung Yeo.

  * The proposed DialOgue Chain-of-ThOught Reasoner (DOCTOR), is a knowledge distillation framework that leverages LLMs as unreliable teachers and selectively distills consistent and helpful rationales via alignment filters. DOCTOR provides reliable CoT rationales for response generation.

* On Complementarity Objectives for Hybrid Retrieval (ACL 2023) - Dohyeon Lee, Seung-won Hwang, Kyungjae Lee, Seungtaek Choi, and Sunghyun Park.

  * The proposed Ratio of Complementarity (RoC), is a new objective which captures a fuller notion of complementarity. Improving RoC of model improves the performance of hybrid retrieval.

* Script, Language, and Labels: Overcoming Three Discrepancies for Low-Resource Language Specialization (AAAI 2023) - Jaeseong Lee, Dohyeon Lee, and Seung-won Hwang.

  * The three discrepancies from Masked Language Modeling (MLM) pretraining, Script, Language, and Labels, lead into a naive specialization as such can be suboptimal. Script and linguistic discrepancy of the target language from the related seen languages, hinder a positive transfer, for which authors propose to maximize representation similarity, unlike existing approaches maximizing overlaps. In addition, label space for MLM prediction can vary across languages, for which authors propose to reinitialize top layers for a more effective adaptation.

* Retrieval-augmented Video Encoding for Instructional Captioning (ACL 2023) - Yeonjoon Jung, Minsoo Kim, Seungtaek Choi, Jihyuk Kim, Minji Seo, and Seung-won Hwang.

  * The proposed retrieval-based framework augments the model representations in the presence of key-object degeneracy. This framework repairs key-object degeneracy, where any single modality fails to sufficiently capture the key objects reffered to in the procedure, in the instructional video.

* Learning to Rank Generation with Pairwise Partial Rewards (EMNLP 2023) - Youngwon Lee, Jinu Lee, and Seung-won Hwang.

  * The proposed reward shaping method provides partial rewards for intermediate actions taken on partial sequences. This method enables the model to promptly prioritize actions that lead to the generation of more desirable sequences.

* Relevance-assisted Generation for Robust Zero-shot Retrieval (EMNLP 2023) - Jihyuk Kim, Minsoo Kim, Joonsuk Park, and Seung-won Hwang.

  * The proposed relevance-guided generation, is divided in two simple subtasks, generating relevance explanations and guiding the generation to avoid negative generalization. Relevance-guided generation method is more robust to domain shifts when key biases cause sampled Psuedo Queries (PQ) to be irrelevant, negatively contributing to generalization. 

