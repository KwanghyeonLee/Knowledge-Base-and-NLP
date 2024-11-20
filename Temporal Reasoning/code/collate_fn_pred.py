import json
import random
import torch
import math
from torch.utils.data import Dataset
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


class GraphDataCollator:
    def __init__(self, device, padding_value=0, evaluation=False,
     contrastive_loss=0, n_negs=0, all_dataset=None, batch_size=1):
        self.padding_value = padding_value
        self.is_eval = evaluation
        self.device = device
        self.n_negs = n_negs # if negative sampling
        if self.n_negs: 
            assert all_dataset is not None
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

class OtrDataset(Dataset):
    def __init__(self, features, batch_size, evaluation=True, full_passage=False):
        base_dir = "../data/contrast_pairs"
        if not os.path.exists(base_dir):
            base_dir = "data/contrast_pairs" # for vscode debug
        prefix = ""
        if evaluation :
            q_rel_list = json.load(open(os.path.join(base_dir, f"test_rel_list{prefix}.json")))
        else:
            exit("this is for test prediction")
        # for faster search
        feature_dict = dict()
        for i, elem in enumerate(features):
            elem['key_indice'] = i
            feature_dict[elem['question_id']] = elem
        
        self.feature_list = []
        if evaluation:
            for qids in q_rel_list:
                self.feature_list.append([feature_dict[qid] for qid in qids])
        
    def __len__(self):
        return len(self.feature_list)
    def __getitem__(self, index):
        return self.feature_list[index]