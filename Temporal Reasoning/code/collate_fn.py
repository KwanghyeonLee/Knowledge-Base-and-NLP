import json
import random
import torch
import math
from torch.utils.data import Dataset
from utils_gnn import depper_map, tagger_map
import torch.nn.functional as F
import os
import itertools
class NegSampler:
    def __init__(self, dataset, n_negative_ctxs, is_eval=True) -> None:
        pairs_dir = "data/contrast_pairs"
        if not os.path.exists(pairs_dir):
            pairs_dir = "../data/contrast_pairs"
        train_qa_pairs_all = json.load(open(os.path.join(pairs_dir,"q_rel_pairs_train.json")))
        dev_qa_pairs_all = json.load(open(os.path.join(pairs_dir,"q_rel_pairs_dev.json")))
        if is_eval:
            self.qa_pairs_all = dev_qa_pairs_all
        else:
            self.qa_pairs_all = train_qa_pairs_all
        self.dataset = dataset
        # for faster search -> mapping
        self.qids = [f[0]['question_id'] for f in dataset]
        indices = [f[1] for f in dataset]
        self.qid_to_indice_map = {self.qids[i]: indices[i]
                                  for i in range(len(indices))}
        self.n_negative_ctxs = n_negative_ctxs
        pass
    def __call__(self, batch):
        _, instance_indices = zip(*batch)
        ctxs = []
        same_p_mask = []
        for i, indice in enumerate(instance_indices):
            qa_pairs = self.qa_pairs_all[self.qids[indice]]['contrast_q']
            if qa_pairs:
                random.shuffle(qa_pairs)
                if self.n_negative_ctxs > -1: # below -1 means use all negatives
                    negative_idxs = qa_pairs[:self.n_negative_ctxs]
                else:
                    negative_idxs = qa_pairs
                try:
                    negative_idxs = [self.qid_to_indice_map[elem['qid']] for elem in negative_idxs]
                except KeyError:
                   negative_idxs = []
            else: #doesn't have contrast questions
                negative_idxs = []
            all_idxs = [indice] + negative_idxs
            same_p_mask.append([len(ctxs)+i for i in range(len(all_idxs))])
            all_ctxs = [self.dataset[i] for i in all_idxs]
            ctxs.extend(all_ctxs)
        return ctxs, same_p_mask

class GraphDataset(Dataset):
    def __init__(self, features, all_key_indices, evaluation=False):
        self.features = features
        self.all_key_indices = all_key_indices

    def __len__(self):
        return len(self.features)
    def __getitem__(self, index):
        return (self.features[index], self.all_key_indices[index])

class GraphDataCollator:
    def __init__(self, device, padding_value=0, evaluation=False,
     contrastive_loss=0, n_negs=0, all_dataset=None, batch_size=1):
        self.padding_value = padding_value
        self.is_eval = evaluation
        self.device = device
        self.n_negs = n_negs # if negative sampling
        if self.n_negs: 
            assert all_dataset is not None
            self.negsampler = NegSampler(all_dataset, n_negative_ctxs=self.n_negs, is_eval=evaluation)
        self.contrastive_loss = bool(contrastive_loss)
        self.batch_size = batch_size

    def __call__(self, batch):
        same_p_mask = None
        if self.contrastive_loss: # dynamic batch style
            features = batch[0]
            # if evaluation: use all sentences
            if not self.is_eval:
                random.shuffle(features)
                # cons: cannot see some instance? -> it will be fine if #epochs is large enough 
                features = features[:self.batch_size]   
            instance_indices = [f['key_indice'] for f in features]
        else:
            if self.n_negs:
                batch, same_p_mask = self.negsampler(batch)
            features, instance_indices = zip(*batch)

        input_ids = [f['input_ids'] for f in features]
        input_masks = [f['mask_ids'] for f in features] 
        segment_ids = [f['segment_ids'] for f in features]
        head_masks = [f['q_head_mask'] for f in features]
        question_len = [f['question_len'] for f in features]
        edges = [f['edges'] for f in features]
        q_tr_word_idx = [f['q_tr_word_idx'] for f in features]
        q_event_word_idx = [f['q_event_word_idx'] for f in features]

        # padding
        pad_len_edges = max([len(elem) for elem in edges])
        #pad_input_ids = max([len(iid) for iid in input_ids])

        # to add parser label info
        #parser_labels = None

        for i in range(len(features)):
            # pad
            edges[i] += [edges[i][0]] * (pad_len_edges - len(edges[i])) # padding this way doesn't create extra edges
            #input_ids[i] += [0] * (pad_input_ids-len(input_ids[i]))
            #input_masks[i] += [0] * (pad_input_ids-len(input_masks[i]))
            #segment_ids[i] += [0] * (pad_input_ids-len(segment_ids[i]))

        edges = torch.tensor(edges, dtype=torch.long).to(self.device)
        input_ids = torch.tensor(input_ids, dtype=torch.long).to(self.device)
        input_masks = torch.tensor(input_masks, dtype=torch.long).to(self.device)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long).to(self.device)

        samples = input_ids, input_masks, segment_ids, instance_indices, head_masks, question_len, edges, \
                    q_tr_word_idx, q_event_word_idx, same_p_mask #parser_labels
        return samples

class QAGNNDataCollator:
    def __init__(self, device, padding_value=0, evaluation=False, use_semi_struct=False, graph_construction = 0):
        self.padding_value = padding_value
        self.is_eval = evaluation
        self.device = device
        self.use_semi_struct = use_semi_struct
        self.graph_construction = graph_construction

    def __call__(self, batch):
        features, instance_indices = zip(*batch)

        input_ids = [f['input_ids'] for f in features]
        input_masks = [f['mask_ids'] for f in features] 
        segment_ids = [f['segment_ids'] for f in features]
        batch_edges = [f['edges'] for f in features]
        bpe_to_node = [f['bpe_to_node'] for f in features]
        pseudo_questions = [f['extra_info'] for f in features]
        if self.use_semi_struct:
            pseudo_questions = [dict(input_ids = q['input_ids'].to(self.device), 
            attention_mask = q['attention_mask'].to(self.device)) if len(q)>0 else dict() for q in pseudo_questions]            
        else:
            pseudo_questions = None
        #default
        input_ids = torch.tensor(input_ids, dtype=torch.long).to(self.device)
        input_masks = torch.tensor(input_masks, dtype=torch.long).to(self.device)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long).to(self.device)
        #question_len = torch.tensor(question_len).to(self.device)
        #added
        #head_masks = torch.tensor(head_masks).to(self.device)
        # to add parser label info
        node_types = []
        node_scores = []
        adj_lengths = []
        edge_indexs = []
        edge_types = []
        pseudo_lens = []
        for bidx, edges in enumerate(batch_edges):
            # https://stackoverflow.com/questions/6473679/transpose-list-of-lists
            head_index, tail_index, edge_type, node_type = [list(i) for i in zip(*edges)]
            edge_indexs.append(torch.tensor([head_index, tail_index]).to(self.device))
            edge_types.append(torch.tensor(edge_type).to(self.device))
            if self.use_semi_struct:
                pseudo_len = pseudo_questions[bidx].get('input_ids', 0)
                if type(pseudo_len) == torch.Tensor:
                    pseudo_len = pseudo_len.size(0)
                pseudo_lens.append(pseudo_len)
                node_type = [0] + [0] * pseudo_len + node_type[:node_type.index(0)] #0 번째 = 겹치지 않는 마지막 node
            elif self.graph_construction == 2:
                node_type = [0] + [1] * len(bpe_to_node[bidx]) #0 번째 = 겹치지 않는 마지막 node
            elif self.graph_construction == 3:
                # node type: 0 = question sum node 1 = question node 2 = passage node
                _node_type_tmp = node_type[:node_type.index(0)]
                n_question_node = len([n for n in _node_type_tmp if n == 1])
                node_type = [0] + [1] * n_question_node + \
                     [2] * (len(bpe_to_node[bidx]) - n_question_node)
            else:
                node_type = [0] + node_type[:node_type.index(0)] #0 번째 = 겹치지 않는 마지막 node
            
            # assert len(list(set(node_type))) == len(node_type) # comment out. type이 같을 필요는 없다
            node_types.append(node_type)
            adj_lengths.append(len(node_type))
        # padding
        max_len_nodes = max([len(elem) for elem in node_types])


        for bidx in range(len(features)):
            # pad
            node_types[bidx] += [node_types[bidx][0]] * (max_len_nodes - len(node_types[bidx])) # padding this way doesn't create extra edges
        node_types = torch.tensor(node_types).to(self.device)
        adj_lengths = torch.tensor(adj_lengths).to(self.device)

        node_scores = torch.ones_like(node_types).unsqueeze(-1).to(self.device)

        samples = input_ids, input_masks, segment_ids, instance_indices,\
                    (node_types, node_scores, adj_lengths, edge_indexs, edge_types), bpe_to_node, (pseudo_questions, pseudo_lens)

        return samples

class OtrDataset(Dataset):
    def __init__(self, features, batch_size, evaluation=False, full_passage=False, train_ratio=1):
        base_dir = "../data/contrast_pairs"
        if not os.path.exists(base_dir):
            base_dir = "data/contrast_pairs" # for vscode debug
        prefix = ""
        if evaluation :
            q_rel_list = json.load(open(os.path.join(base_dir, f"dev_rel_list{prefix}.json")))
        else:
            q_rel_list = json.load(open(os.path.join(base_dir, f"train_rel_list{prefix}.json")))
        # call question pseudo-labels
        question_labels = "../question_analysis/question_labels.json"
        if not os.path.exists(question_labels):
            question_labels = "question_analysis/question_labels.json" # for vscode debug
        question_labels = json.load(open(question_labels))
        # for faster search
        feature_dict = dict()
        for i, elem in enumerate(features):
            elem['key_indice'] = i
            elem['question_labels'] = question_labels[elem['question_id']]["labels"]
            feature_dict[elem['question_id']] = elem
        
        self.feature_list = []
        if evaluation:
            for qids in q_rel_list:
                self.feature_list.append([feature_dict[qid] for qid in qids])
        else:
            if train_ratio < 1:
                for qids in q_rel_list:
                    for _ in range(math.ceil(len(qids) / batch_size)):  # ex. 27 instances in one cluster -> ceil(27/6) = 5 same instances
                        to_append_list = [feature_dict[qid] for qid in qids if qid in feature_dict]
                        if len(to_append_list) > 0:
                            self.feature_list.append(to_append_list)
            else:
                for qids in q_rel_list:
                    for _ in range(math.ceil(len(qids) / batch_size)):  # ex. 27 instances in one cluster -> ceil(27/6) = 5 same instances
                        self.feature_list.append([feature_dict[qid] for qid in qids])

    def __len__(self):
        return len(self.feature_list)
    def __getitem__(self, index):
        return self.feature_list[index]

class OtrDataCollator:
    def __init__(self, device, batch_size, evaluation=False):
        self.batch_size = batch_size
        self.is_eval = evaluation
        self.device = device
        self.evaluation=evaluation

    def __call__(self, batch):
        batch = batch[0]
        # if evaluation: use all sentences
        if not self.evaluation:
            random.shuffle(batch)
            # cons: cannot see some instance? -> it will be fine if #epochs is large enough 
            batch = batch[:self.batch_size]

        input_ids = [f['input_ids'] for f in batch]
        input_masks = [f['mask_ids'] for f in batch] 
        segment_ids = [f['segment_ids'] for f in batch]
        instance_indices = [f['key_indice'] for f in batch]
        offsets = [f['offset'] for f in batch]
        question_mask = [f['question_mask'] for f in batch]
        question_labels = [f['question_labels'] for f in batch]
        labels = [f['label'] for f in batch]

        input_ids = torch.tensor(input_ids, dtype=torch.long).to(self.device)
        input_masks = torch.tensor(input_masks, dtype=torch.long).to(self.device)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long).to(self.device)
        labels = torch.tensor(labels, dtype=torch.long).to(self.device)
        question_labels = torch.tensor(question_labels, dtype=torch.float32).to(self.device)
        question_mask = torch.tensor(question_mask, dtype=torch.bool).to(self.device)

        samples = input_ids, input_masks, segment_ids, instance_indices, offsets, question_mask, question_labels, labels

        return samples


class OtrGraphDataset(Dataset):
    def __init__(self, features, batch_size, evaluation=False, full_passage=False):
        base_dir = "../data/contrast_pairs"
        if not os.path.exists(base_dir):
            base_dir = "data/contrast_pairs" # for vscode debug
        prefix = "train"
        if evaluation :
            prefix = "dev"
        q_rel_list = json.load(open(os.path.join(base_dir, f"in_p_q_pairs_{prefix}.json")))
        # for faster search
        feature_dict = dict()
        for i, elem in enumerate(features):
            elem['key_indice'] = i
            feature_dict[elem['question_id']] = elem
        
        self.feature_list = []
        for _, qid_lists in q_rel_list.items():
            feature_list_agg = []
            for qids_list in qid_lists:
                feature_list_agg.append([feature_dict[qid] for qid in qids_list])
            self.feature_list.append(feature_list_agg)

    def __len__(self):
        return len(self.feature_list)
    def __getitem__(self, index):
        return self.feature_list[index]

#TODO: i fixed this. So qagnn will not work
class GraphDataCollatorCont:
    def __init__(self, device, batch_size, evaluation=False):
        self.device = device
        self.batch_size = batch_size
        self.evaluation=evaluation

    def __call__(self, batch):
        #features, instance_indices, event_indices = zip(*batch)
        batch = batch[0]
        # if evaluation: use all sentences
        if not self.evaluation:
            #random.shuffle(batch)
            for subbatch in batch:
                random.shuffle(subbatch)
        batch.sort(key=len, reverse=True)
        _cluster_lengths = [len(s) for s in batch]
        cluster_lengths_accum = [0]
        for i in range(0,len(_cluster_lengths)):
            cluster_lengths_accum.append(cluster_lengths_accum[i]+_cluster_lengths[i])
        cluster_lengths_accum = cluster_lengths_accum[1:]
            
        batch = list(itertools.chain.from_iterable(batch))
        # batch안에 크기들을 표시해서, 6 넘기면 거기서 cut 해보자.
        p_samples = [] # samples containing all qs from p
        sub_idx = 0
        if self.evaluation:
            self.batch_size = len(batch)
        while sub_idx < len(batch):
            if sub_idx + self.batch_size <= len(batch):
                last_sub_idx = sub_idx + self.batch_size
                if last_sub_idx + 1 in cluster_lengths_accum:
                    last_sub_idx += 1
                elif last_sub_idx + 2 in cluster_lengths_accum:
                    last_sub_idx += 2
            else:
                last_sub_idx = len(batch)
            
            sub_batch = batch[sub_idx : last_sub_idx]
            input_ids = [f['input_ids'] for f in sub_batch]
            input_masks = [f['mask_ids'] for f in sub_batch] 
            segment_ids = [f['segment_ids'] for f in sub_batch]
            head_masks = [f['q_head_mask'] for f in sub_batch]
            question_len = [f['question_len'] for f in sub_batch]
            instance_indices = [f['key_indice'] for f in sub_batch]
            edges = [f['edges'] for f in sub_batch]
            q_tr_word_idx = [f['q_tr_word_idx'] for f in sub_batch]
            q_event_word_idx = [f['q_event_word_idx'] for f in sub_batch]

            input_ids = torch.tensor(input_ids, dtype=torch.long).to(self.device)
            input_masks = torch.tensor(input_masks, dtype=torch.long).to(self.device)
            segment_ids = torch.tensor(segment_ids, dtype=torch.long).to(self.device)

            # padding
            pad_len_edges = max([len(elem) for elem in edges])

            for i in range(len(sub_batch)):
                # pad
                edges[i] += [edges[i][0]] * (pad_len_edges - len(edges[i])) # padding this way doesn't create extra edges

            same_p_mask = None

            edges = torch.tensor(edges, dtype=torch.long).to(self.device)

            samples = input_ids, input_masks, segment_ids, instance_indices, head_masks, question_len, edges, \
                        q_tr_word_idx, q_event_word_idx, same_p_mask
            p_samples.append(samples)

            sub_idx += len(sub_batch)

        return p_samples