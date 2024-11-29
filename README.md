# Knowledge-Base-and-NLP

#### This repository organizes researches related to AI Technology Development for Commonsense Extraction, Reasoning, and Inference from Heterogeneous Data, especially Knowledge-Base-and-NLP task.
#### This repository summarizes following researches.

## Research list
    
* CoTEVer: Chain of Thought Prompting Annotation Toolkit for Explanation Verification (EACL 2023) - Seungone Kim, Se June Joo, Yul Jang, Hyungjoo Chae, and Jinyoung Yeo.

  * The proposed Chain of Thought Prompting Annotation Toolkit for Explanation Verification (CoTEVer), is a tool-kit for annotating the factual correctness of generated explanations and collecting revision data of wrong explanations.

* COMMIT: Code-Mixing English-Centric Large Language Model for Multilingual Instruction Tuning (Findings of NAACL 2024) - Jaeseong Lee, YeonJoon Jung, and Seung-won Hwang.

  * The proposed code-mixed continual causal language modeling to align the decoder improves the exact match score of low-resourced language QA task by up to 32x.

* Mind the Gap! Injecting Commonsense Knowledge for Abstractive Dialogue Summarization (COLING 2022) - Seungone Kim, Se June Joo, Hyungjoo Chae, Chaehyeong Kim, Seung-won Hwang, and Jinyoung Yeo.

  * The proposed Summarizing with Injected Commonsense Knowledge (SICK), is a framework that uses commonsense inferences as additional context. SICK leverages the unique characteristics of dialogues sharing commonsense knowledge across participants, to resolve the difficulties in summarizing them.

* ContrastiveMix: Overcoming Code-Mixing Dilemma in Cross-Lingual Transfer for Information Retrieval (NAACL 2024) - Junggeun Do, Jaeseong Lee, and Seung-won Hwang.

  * The proposed ContrastiveMix balances the tension between the positive effect of code-mixing on aligning representations across languages and the negative impact it has on IR-specific objective of matching representations between queries and relevant passages.

* Dialogue Chain-of-Thought Distillation for Commonsense-aware Conversational Agents (EMNLP 2023) - Hyungjoo Chae, Yongho Song, Kai Tzu-iunn Ong, Taeyoon Kwon, Minjin Kim, Youngjae Yu, Dongha Lee, Dongyeop Kang, and Jinyoung Yeo.

  * The proposed DialOgue Chain-of-ThOught Reasoner (DOCTOR), is a knowledge distillation framework that leverages LLMs as unreliable teachers and selectively distills consistent and helpful rationales via alignment filters. DOCTOR provides reliable CoT rationales for response generation.

* DADA: Distribution-Aware Domain Adaptation of PLMs for Information Retrieval (Findings of ACL 2024) - Dohyeon Lee, Jongyoon Kim, Seung-won Hwang and Joonsuk Park.

  * The proposed DADA tackles the failure of pseudo-query generation for domain adaptation of informration retrieval in resembling real queries in the target domain, by incorporating term distirbution feedback.

* On Complementarity Objectives for Hybrid Retrieval (ACL 2023) - Dohyeon Lee, Seung-won Hwang, Kyungjae Lee, Seungtaek Choi, and Sunghyun Park.

  * The proposed Ratio of Complementarity (RoC), is a new objective which captures a fuller notion of complementarity. Improving RoC of model improves the performance of hybrid retrieval.

* Script-mix: Mixing Scripts for Low-resource Language Parsing (NAACL 2024) - Jaeseong Lee, Dohyeon Lee, and Seung-won Hwang.

  * The proposed ScriptMix, combines the complementary strengths and overcomes the hurdle in realizing the integration of the two, transliteration and vocabulary augmentation, for low-resource language adaptation of multilinugal pretrained language models.

* Script, Language, and Labels: Overcoming Three Discrepancies for Low-Resource Language Specialization (AAAI 2023) - Jaeseong Lee, Dohyeon Lee, and Seung-won Hwang.

  * The three discrepancies from Masked Language Modeling (MLM) pretraining, Script, Language, and Labels, lead into a naive specialization as such can be suboptimal. Script and linguistic discrepancy of the target language from the related seen languages, hinder a positive transfer, for which authors propose to maximize representation similarity, unlike existing approaches maximizing overlaps. In addition, label space for MLM prediction can vary across languages, for which authors propose to reinitialize top layers for a more effective adaptation.

* Retrieval-augmented Video Encoding for Instructional Captioning (ACL 2023) - Yeonjoon Jung, Minsoo Kim, Seungtaek Choi, Jihyuk Kim, Minji Seo, and Seung-won Hwang.

  * The proposed retrieval-based framework augments the model representations in the presence of key-object degeneracy. This framework repairs key-object degeneracy, where any single modality fails to sufficiently capture the key objects reffered to in the procedure, in the instructional video.

* Learning to Rank Generation with Pairwise Partial Rewards (EMNLP 2023) - Youngwon Lee, Jinu Lee, and Seung-won Hwang.

  * The proposed reward shaping method provides partial rewards for intermediate actions taken on partial sequences. This method enables the model to promptly prioritize actions that lead to the generation of more desirable sequences.

* Relevance-assisted Generation for Robust Zero-shot Retrieval (EMNLP 2023) - Jihyuk Kim, Minsoo Kim, Joonsuk Park, and Seung-won Hwang.

  * The proposed relevance-guided generation, is divided in two simple subtasks, generating relevance explanations and guiding the generation to avoid negative generalization. Relevance-guided generation method is more robust to domain shifts when key biases cause sampled Psuedo Queries (PQ) to be irrelevant, negatively contributing to generalization. 

* Inductive Reasoning Elicitation for Temporal Relation Understanding (EACL 2024) - Jongho Kim, Dohyeon Lee, Minsoo Kim, and Seung-won Hwang.

  * The proposed TRN (timeline reasoning network) outperforms previous methods for temporal reading comprehension and temporal relation extraction tasks, by effectively resolving the spurious overlaps in answers using the predicted timeline.

* Train-Attention: Meta-Learning Where to Focus in Continual Knowledge Learning (NeurIPS 2024) - Yeongbin Seo, Dongha Lee, and Jinyoung Yeo.

  * The proposed Train-Attention, is a meta-learning framework that adaptively adjusts the attention mechanism during training to focus on relevant tokens, thereby improving the model's ability to learn new information without forgetting previously acquired knowledge.

* Large Language Models Are Clinical Reasoners: Reasoning-Aware Diagnosis Framework with Prompt-Generated Rationales (AAAI 2024) - Taeyoon Kwon, Kai Tzu-iunn Ong, Dongjin Kang, Seungjun Moon, Jeong Ryong Lee, Dosik Hwang, Beomseok Sohn, Yongsik Sim, Dongha Lee, and Jinyoung Yeo.

  * The proposed reasoning-aware diagnosis framework, is a novel framework that leverages LLMs as unreliable teachers and selectively distills consistent and helpful rationales via alignment filters.

* Language Models as Compilers (EMNLP 2024) - Hyungjoo Chae, Yeonghyeon Kim, Seungone Kim, Kai Tzu-iunn Ong, Beong-woo Kwak, Moohyeon Kim, Seonghwan Kim, Taeyoon Kwon, Jiwan Chung, Youngjae Yu, and Jinyoung Yeo.

  * The proposed language models as compilers, is a novel framework that uses psuedocode to have a better control over LLMs' reasoning process.

* Can Large Language Models be Good Emotional Supporter? Mitigating Preference Bias on Emotional Support Conversation (ACL 2024) - Dongjin Kang, Sunghwan Kim, Taeyoon Kwon, Seungjun Moon, Hyunsouk Cho, Youngjae Yu, Dongha Lee, and Jinyoung Yeo.

  * This paper first investigates the ability of LLMs as emotional supporters, and then proposes a method to mitigate the preference bias of LLMs on emotional support conversations.

* Cactus: Towards Psychological Counseling Conversations using Cognitive Behavioral Theory (EMNLP 2024) - Suyeon Lee, Sunghwan Kim, Minju Kim, Dongjin Kang, Dongil Yang, Harim Kim, Minseok Kang, Dayi Jung, Min Hee Kim, Seungbeen Lee, Kyoung-Mee Chung, Youngjae Yu, Dongha Lee, and Jinyoung Yeo.

  * This paper proposes a psychological counseling conversation framework using cognitive behavioral theory. In addition, it provides a new dataset for psychological counseling conversation.

* Commonsense-augmented Memory Construction and Management in Long-term Conversations via Context-aware Persona Refinement (EACL 2024) - Hana Kim, Kai Tzu-iunn Ong, Seoyeon Kim, Dongha Lee, and Jinyoung Yeo.

  * This paper introduces a novel framework that leverages commonsense-based persona expansion to refine uninformative persona sentences in long-term conversations, enhancing response quality through human-like persona refinement.

* Multitask Deep Learning for Joint Detection of Necrotizing Viral and Noninfectious Retinitis From Common Blood and Serology Test Data (IOVS 2024) - Kai Tzu-iunn Ong, Taeyoon Kwon, Harok Jang, Min Kim, Christopher Seungkyu Lee, Suk Ho Byeon, Sung Soo Kim, Jinyoung Yeo, Eun Young Choi.

  * This paper proposes a multitask deep learning framework for joint detection of necrotizing viral and noninfectious retinitis from common blood and serology test data.

* Coffee-Gym: An Environment for Learning Feedback for Code Editing (EMNLP 2024) - Hyungjoo Chae, Taeyoon Kwon, Seungjun Moon, Yongho Song, Dongjin Kang, Kai Tzu-iunn Ong, Beong-woo Kwak, Seonghyeon Bae, Seung-won Hwang, and Jinyoung Yeo.

  * This paper presents Coffee-Gym, a comprehensive RL environment for training models that provide feedback on code editing.
