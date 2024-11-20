from typing import Iterator, List, Mapping, Union, Optional, Set
from collections import defaultdict, Counter, OrderedDict
from datetime import datetime
import json
from tqdm import tqdm
import pickle
import numpy as np
import logging
import spacy
import torch
from glob import glob
import random
import os
import string
from copy import deepcopy

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
nlp = spacy.load("en_core_web_lg")
tagger_map = {elem : i+1 for i, elem in enumerate(nlp.get_pipe('tagger').labels)}
tagger_map["_SP"] = max(tagger_map.values())+1
depper_map = {elem : i+2 for i, elem in enumerate(nlp.get_pipe('parser').labels)}

def load_data(data_dir, split, suffix, train_ratio=1):
    filename = "%s%s%s" % (data_dir, split, suffix)
    print("==========load data from %s ===========" % filename)
    with open(filename, "r") as read_file:
        return_file = json.load(read_file)
        #TODO: erase this when debug done!
        if train_ratio<1.0:
            return dict(random.sample(return_file.items(), int(len(return_file) * train_ratio)))
        else:
            return return_file

class ConfigurationError(Exception):
    """
    The exception raised by any AllenNLP object when it's misconfigured
    (e.g. missing properties, invalid properties, unknown properties).
    """

    def __init__(self, message):
        super(ConfigurationError, self).__init__()
        self.message = message

    def __str__(self):
        return repr(self.message)

def replace_masked_values(tensor: torch.Tensor, mask: torch.Tensor, replace_with: float) -> torch.Tensor:
    """
    Replaces all masked values in ``tensor`` with ``replace_with``.  ``mask`` must be broadcastable
    to the same shape as ``tensor``. We require that ``tensor.dim() == mask.dim()``, as otherwise we
    won't know which dimensions of the mask to unsqueeze.

    This just does ``tensor.masked_fill()``, except the pytorch method fills in things with a mask
    value of 1, where we want the opposite.  You can do this in your own code with
    ``tensor.masked_fill((1 - mask).byte(), replace_with)``.
    """
    if tensor.dim() != mask.dim():
        raise ConfigurationError("tensor.dim() (%d) != mask.dim() (%d)" % (tensor.dim(), mask.dim()))
    return tensor.masked_fill((1 - mask).bool(), replace_with)


def select_field(data, field):
    # collect a list of field in data                                                                  
    # fields: 'label', 'offset', 'input_ids, 'mask_ids', 'segment_ids', 'question_id'                            
    return [ex[field] for ex in data]

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def exact_match(question_ids, labels, predictions):
    em = defaultdict(list)
    for q, l, p in zip(question_ids, labels, predictions):
        em[q].append(l == p)
    print("Total %s questions" % len(em))
    return float(sum([all(v) for v in em.values()])) / float(len(em))

def cal_f1(pred_labels, true_labels, label_map, log=False):
    def safe_division(numr, denr, on_err=0.0):
        return on_err if denr == 0.0 else numr / denr

    assert len(pred_labels) == len(true_labels)

    num_tests = len(true_labels)

    total_true = Counter(true_labels)
    total_pred = Counter(pred_labels)

    labels = list(label_map)

    n_correct = 0
    n_true = 0
    n_pred = 0

    label_map
    # we only need positive f1 score
    exclude_labels = ['Negative']
    for label in labels:
        if label not in exclude_labels:
            true_count = total_true.get(label, 0)
            pred_count = total_pred.get(label, 0)

            n_true += true_count
            n_pred += pred_count

            correct_count = len([l for l in range(len(pred_labels))
                                 if pred_labels[l] == true_labels[l] and pred_labels[l] == label])
            n_correct += correct_count
    if log:
        logger.info("Correct: %d\tTrue: %d\tPred: %d" % (n_correct, n_true, n_pred))
    precision = safe_division(n_correct, n_pred)
    recall = safe_division(n_correct, n_true)
    f1_score = safe_division(2.0 * precision * recall, precision + recall)
    if log:
        logger.info("Overall Precision: %.4f\tRecall: %.4f\tF1: %.4f" % (precision, recall, f1_score))
    return f1_score

def cal_pre_recall(pred_labels, true_labels, label_map, log=False):
    def safe_division(numr, denr, on_err=0.0):
        return on_err if denr == 0.0 else numr / denr

    assert len(pred_labels) == len(true_labels)

    num_tests = len(true_labels)

    total_true = Counter(true_labels)
    total_pred = Counter(pred_labels)

    labels = list(label_map)

    n_correct = 0
    n_true = 0
    n_pred = 0

    # we only need positive f1 score
    exclude_labels = ['Negative']
    for label in labels:
        if label not in exclude_labels:
            true_count = total_true.get(label, 0)
            pred_count = total_pred.get(label, 0)

            n_true += true_count
            n_pred += pred_count

            correct_count = len([l for l in range(len(pred_labels))
                                 if pred_labels[l] == true_labels[l] and pred_labels[l] == label])
            n_correct += correct_count
    precision = safe_division(n_correct, n_pred)
    recall = safe_division(n_correct, n_true)
    return precision, recall

def sample_errors(passages, questions, answers, labels, preds, label_class="Positive", num=50):
    assert len(passages) == len(preds)
    assert len(questions) == len(preds)
    assert len(answers) == len(preds)
    errors = []
    outfile = open("%s_error_samples.tsv" % label_class, 'w')
    outfile.write("Passage\tQuestion\tAnswer-span\tAnswer-offset\tAnswer-label\tAnswer-prediction\n")
    count = 0
    for pa, q, a, l, p in zip(passages, questions, answers, labels, preds):
        if count >= num:
            continue
        if l == label_class and l != p:
            outfile.write("%s\t%s\t%s\t%s\t%s\t%s\n" % (" ".join(pa), q, a['span'], a['idx'], l, p))
            count += 1
    outfile.close()
    return

def get_train_dev_ids(data_dir, data_type):
    trainIds = [f.strip() for f in open("%s/%s/trainIds.txt" % (data_dir, data_type))]
    devIds = [f.strip() for f in open("%s/%s/devIds.txt" % (data_dir, data_type))]
    return trainIds, devIds

def find_root_token(tok):
    root = tok
    while root.i != root.head.i:
        root = root.head
    return root.i


#TODO: max question length, align for BERT, 
def convert_to_features_bert_graph2(data, tokenizer, max_length=150, evaluation=False, max_question_length = 0,
                        instance=True, end_to_end=False, graph_construction = 0, train_ratio = 1.0):
    # each sample will have [CLS] + Question + [SEP] + Context                                                                                 
    print(f"train_{tokenizer.__class__.__name__}_{graph_construction}_{train_ratio}.pkl")
    if not evaluation and os.path.exists(f"train_{tokenizer.__class__.__name__}_{graph_construction}_{train_ratio}.pkl"):
        print("load data from pickle. If you revised data reader, delete this script")
        samples =pickle.load(open(f"train_{tokenizer.__class__.__name__}_{graph_construction}_{train_ratio}.pkl", 'rb'))
        return samples    
    samples = []
    counter = 0
    max_len_global = 0 # to show global max_len without truncating                                                                             

    def get_edges(doc):
        if graph_construction == 0:
            edges = [(token.i, token.head.i, int(token.dep_=="ROOT"), token) for token in doc]
        elif graph_construction == 1:
            edges = [(token.i, token.head.i, int(token.dep_=="ROOT" or token.pos_=="VERB"), token) for token in doc]
        else: 
            edges = [(token.i, token.head.i, int(token.dep_=="ROOT"), token) for token in doc]
        return edges

    for k, v in tqdm(data.items(), desc = "spacy & preprocess"):
        counter += 1
        #if counter<=6400: continue
        if train_ratio < 1 and counter > len(data) * train_ratio:
            continue
        segment_ids = []
        start_token = ['[CLS]']
        v['question'] = v['question'].strip()
        question = tokenizer.tokenize(v['question'])
        q_doc = nlp(v['question'])
        q_edges = get_edges(q_doc)
        # the following bert tokenized context starts / end with ['SEP']                                                                       
        new_tokens = ["[SEP]"]
        orig_to_tok_map = []
        for i, token in enumerate(v['context']):
            orig_to_tok_map.append(len(new_tokens))
            temp_tokens = tokenizer.tokenize(token)
            new_tokens.extend(temp_tokens)

        new_tokens.append("[SEP]")
        length = len(new_tokens)
        orig_to_tok_map.append(length)
        assert len(orig_to_tok_map) == len(v['context']) + 1 # account for ending ['SEP']                                                      

        # following the bert convention for calculating segment ids                                                                            
        segment_ids = [0] * (len(question) + 2) + [1] * (len(new_tokens) - 1)

        # mask ids                                                                                                                             
        mask_ids = [1] * len(segment_ids)

        tokenized_ids = tokenizer.convert_tokens_to_ids(start_token + question + new_tokens)
        assert len(tokenized_ids) == len(segment_ids)

        if len(tokenized_ids) > max_len_global:
            max_len_global = len(tokenized_ids)

        # truncate long sequence, but we can simply set max_length > global_max                                                                
        if len(tokenized_ids) > max_length:
            ending = tokenized_ids[-1]
            tokenized_ids = tokenized_ids[:-(len(tokenized_ids) - max_length + 1)] + [ending]
            assert len(tokenized_ids) == max_length

        # padding                                                                                                                              
        if len(tokenized_ids) < max_length:
            # Zero-pad up to the sequence length.                                                                                              
            padding = [0] * (max_length - len(tokenized_ids))
            tokenized_ids += padding
            mask_ids += padding
            segment_ids += padding

        if graph_construction == 0 or graph_construction == 1: 
            p_doc = nlp(" ".join(v['context']))
            # for graph construction
            p_edges = get_edges(p_doc)
            q_edges = [(elem[0], elem[1], elem[2]) for elem in q_edges] 

            # bpe와 spacy를 align 하겠다
            _alignment = align_bert_to_words(tokenizer, new_tokens[1:-1], p_doc)
            # bpe 첫번째만 사용하겠다. 앞의 <s> question </s> </s> length 생각해서.
            bpe_to_node = [align_idx[0] + len(question) + 2 for align_idx in _alignment]
            # question도 align
            _alignment_q = align_bert_to_words(tokenizer, question, q_doc)
            bpe_to_node = [align_idx[0] + 1 for align_idx in _alignment_q] + bpe_to_node

            pos_labels = [tagger_map[token.tag_] for token in p_doc]
            dep_labels = [depper_map[token.dep_] for token in p_doc]
            parser_tag_map = [p_edge[0] for p_edge in p_edges]

            # add question length for numgnn
            p_edges = [(elem[0]+max_question_length, elem[1]+max_question_length, elem[2]) for elem in p_edges] 

            edges = q_edges + p_edges
            assert len(bpe_to_node) == len(edges)
            
            verb_edges = []
            len_edges = len(edges)
            for edge_idx1 in range(len_edges):
                for edge_idx2 in range(edge_idx1+1, len_edges):
                    if edges[edge_idx1][2] == 1 and edges[edge_idx2][2] == 1:
                        verb_edges.append((edges[edge_idx1][0], edges[edge_idx2][0], 1))  
                        verb_edges.append((edges[edge_idx2][0], edges[edge_idx1][0], 1))            
            edges += verb_edges

        elif graph_construction == 2: # same mention
            p_doc = nlp(" ".join(v['context']))
            # for graph construction
            p_edges = get_edges(p_doc)

            # bpe와 spacy를 align 하겠다
            _alignment = align_bert_to_words(tokenizer, new_tokens[1:-1], p_doc)
            # bpe 첫번째만 사용하겠다. 앞의 <s> question </s> </s> length 생각해서.
            bpe_to_node = [align_idx[0] + len(question) + 2 for align_idx in _alignment]
            # question도 align
            _alignment_q = align_bert_to_words(tokenizer, question, q_doc)
            bpe_to_node = [align_idx[0] + 1 for align_idx in _alignment_q] + bpe_to_node

            pos_labels = [tagger_map[token.tag_] for token in p_doc]
            dep_labels = [depper_map[token.dep_] for token in p_doc]
            parser_tag_map = [p_edge[0] for p_edge in p_edges]

            # add question length for numgnn
            p_edges = [(elem[0]+max_question_length, elem[1]+max_question_length, elem[2], elem[3]) for elem in p_edges] 

            edges = q_edges + p_edges
            assert len(bpe_to_node) == len(edges)
            
            # 지금 문제: 다른 sentence에 있는 건 못본다?

            other_edges = []
            for edge_idx1 in range(len(edges)):
                # https://machinelearningknowledge.ai/tutorial-on-spacy-part-of-speech-pos-tagging/                
                for edge_idx2 in range(edge_idx1+1, len(edges)):
                    #if edge_idx1 == edge_idx2: continue                    
                    # connect with sentences
                    if edges[edge_idx1][3].dep_ == 'ROOT' and edges[edge_idx2][3].dep_ == 'ROOT':
                        other_edges.append((edges[edge_idx1][0], edges[edge_idx2][0]))
                        other_edges.append((edges[edge_idx2][0], edges[edge_idx1][0]))                
                    if edges[edge_idx1][3].lemma_ == edges[edge_idx2][3].lemma_ and \
                        edges[edge_idx1][3].pos_ in ["NOUN", "VERB", "PRON", "PROPN", "ADJ", "AUX", "NUM"] :
                        other_edges.append((edges[edge_idx1][0], edges[edge_idx2][0]))
                        other_edges.append((edges[edge_idx2][0], edges[edge_idx1][0]))

            edges = [(edge[0], edge[1]) for edge in edges] + other_edges

        if end_to_end:
            labels, offsets = [], []
            events = []
            for kk, vv in enumerate(v['answers']['labels']):
                labels.append(vv)
                offsets.append(orig_to_tok_map[kk] + len(question) + 1)
            for vv in v['answers']['types']:
                events.append(vv)
            sample = {'label': labels,
                      'events': events,
                      'offset': offsets,
                      'input_ids': tokenized_ids,
                      'mask_ids': mask_ids,
                      'segment_ids': segment_ids,
                      'question_id': k,
                      'q_head_mask' : bpe_to_node,
                      'question_len' : len(_alignment_q),
                      "edges": edges,
                      "q_event_word_idx" : [],
                      "q_tr_word_idx" : [],
                      "pos_labels": pos_labels,
                      "dep_labels": dep_labels,
                      "parser_tag_map" : parser_tag_map
                      }
            # add these three field for qualitative analysis                                                                
            if evaluation:
                sample['passage'] = v['context']
                sample['question'] = v['question']
                sample['question_cluster'] = v['question_cluster']
                sample['cluster_size'] = v['cluster_size']
                sample['answer'] = v['answers']
                sample['individual_answers'] = [a['labels'] for a in v['individual_answers']]
            samples.append(sample)
        else:
            # no duplicate P + Q                                                  
            labels, offsets = [], []
            events = []
            for vv in v['answers'].values():
                labels.append(vv['label'])
                offsets.append(orig_to_tok_map[vv['idx']] + len(question) + 1)
            for vv in v['answers']['types']:
                events.append(vv)
            sample = {'label': labels,
                      'events': events,
                      'offset': offsets,
                      'input_ids': tokenized_ids,
                      'mask_ids': mask_ids,
                      'segment_ids': segment_ids,
                      'question_id': k,
                      'q_head_mask' : bpe_to_node,
                      'question_len' : len(q_edges),
                      "edges": edges,
                      "q_event_word_idx" : [],
                      "q_tr_word_idx" : [],
                      "pos_labels": pos_labels,
                      "dep_labels": dep_labels,
                      "parser_tag_map" : parser_tag_map
                      }

            # add these three field for qualitative analysis            
            if evaluation:
                sample['passage'] = v['context']
                sample['question'] = v['question']
                sample['answer'] = v['answers']
                sample['question_cluster'] = v['question_cluster']
                sample['cluster_size'] = v['cluster_size']
                individual_answers = []
                for vv in v['individual_answers']:
                    individual_answers.append([a['label'] for a in vv.values()])
                sample['individual_answers'] = individual_answers
            samples.append(sample)
            
        # check some example data                                                                                                              
        if counter < 0:
            print(k)
            print(v)
            print(tokenized_ids)
        #counter += 1

    print("Maximum length after tokenization is: % s" % (max_len_global))

    if not evaluation:
        pickle.dump(samples, open(f"train_{tokenizer.__class__.__name__}_{graph_construction}_{train_ratio}.pkl", 'wb'))

    return samples


def convert_to_features_bert_graph2_no_label(data, tokenizer, max_length=150, max_question_length = 0, 
                                             evaluation = False, graph_construction = 0):
    # each sample will have [CLS] + Question + [SEP] + Context                                                                                 
    samples = []
    counter = 0
    max_len_global = 0 # to show global max_len without truncating                                                                             

    def get_edges(doc):
        if graph_construction == 0:
            edges = [(token.i, token.head.i, int(token.dep_=="ROOT"), token) for token in doc]
        elif graph_construction == 1:
            edges = [(token.i, token.head.i, int(token.dep_=="ROOT" or token.pos_=="VERB"), token) for token in doc]
        else: 
            edges = [(token.i, token.head.i, int(token.dep_=="ROOT"), token) for token in doc]
        return edges

    for k, v in tqdm(data.items(), desc = "spacy & preprocess"):
        segment_ids = []
        start_token = ['[CLS]']
        v['question'] = v['question'].strip()
        question = tokenizer.tokenize(v['question'])
        q_doc = nlp(v['question'])
        q_edges = get_edges(q_doc)
        # the following bert tokenized context starts / end with ['SEP']                                                                       
        new_tokens = ["[SEP]"]
        orig_to_tok_map = []
        for i, token in enumerate(v['context']):
            orig_to_tok_map.append(len(new_tokens))
            temp_tokens = tokenizer.tokenize(token)
            new_tokens.extend(temp_tokens)

        new_tokens.append("[SEP]")
        length = len(new_tokens)
        orig_to_tok_map.append(length)
        assert len(orig_to_tok_map) == len(v['context']) + 1 # account for ending ['SEP']                                                      

        # following the bert convention for calculating segment ids                                                                            
        segment_ids = [0] * (len(question) + 2) + [1] * (len(new_tokens) - 1)

        # mask ids                                                                                                                             
        mask_ids = [1] * len(segment_ids)

        tokenized_ids = tokenizer.convert_tokens_to_ids(start_token + question + new_tokens)
        assert len(tokenized_ids) == len(segment_ids)

        if len(tokenized_ids) > max_len_global:
            max_len_global = len(tokenized_ids)

        # truncate long sequence, but we can simply set max_length > global_max                                                                
        if len(tokenized_ids) > max_length:
            ending = tokenized_ids[-1]
            tokenized_ids = tokenized_ids[:-(len(tokenized_ids) - max_length + 1)] + [ending]
            assert len(tokenized_ids) == max_length

        # padding                                                                                                                              
        if len(tokenized_ids) < max_length:
            # Zero-pad up to the sequence length.                                                                                              
            padding = [0] * (max_length - len(tokenized_ids))
            tokenized_ids += padding
            mask_ids += padding
            segment_ids += padding


        if graph_construction == 2: # same mention
            p_doc = nlp(" ".join(v['context']))
            # for graph construction
            p_edges = get_edges(p_doc)

            # bpe와 spacy를 align 하겠다
            _alignment = align_bert_to_words(tokenizer, new_tokens[1:-1], p_doc)
            # bpe 첫번째만 사용하겠다. 앞의 <s> question </s> </s> length 생각해서.
            bpe_to_node = [align_idx[0] + len(question) + 2 for align_idx in _alignment]
            # question도 align
            _alignment_q = align_bert_to_words(tokenizer, question, q_doc)
            bpe_to_node = [align_idx[0] + 1 for align_idx in _alignment_q] + bpe_to_node

            pos_labels = [tagger_map[token.tag_] for token in p_doc]
            dep_labels = [depper_map[token.dep_] for token in p_doc]
            parser_tag_map = [p_edge[0] for p_edge in p_edges]

            # add question length for numgnn
            p_edges = [(elem[0]+max_question_length, elem[1]+max_question_length, elem[2], elem[3]) for elem in p_edges] 

            edges = q_edges + p_edges
            assert len(bpe_to_node) == len(edges)
            
            # 지금 문제: 다른 sentence에 있는 건 못본다?

            other_edges = []
            for edge_idx1 in range(len(edges)):
                # https://machinelearningknowledge.ai/tutorial-on-spacy-part-of-speech-pos-tagging/                
                for edge_idx2 in range(edge_idx1+1, len(edges)):
                    #if edge_idx1 == edge_idx2: continue                    
                    # connect with sentences
                    if edges[edge_idx1][3].dep_ == 'ROOT' and edges[edge_idx2][3].dep_ == 'ROOT':
                        other_edges.append((edges[edge_idx1][0], edges[edge_idx2][0]))
                        other_edges.append((edges[edge_idx2][0], edges[edge_idx1][0]))                
                    if edges[edge_idx1][3].lemma_ == edges[edge_idx2][3].lemma_ and \
                        edges[edge_idx1][3].pos_ in ["NOUN", "VERB", "PRON", "PROPN", "ADJ", "AUX", "NUM"] :
                        other_edges.append((edges[edge_idx1][0], edges[edge_idx2][0]))
                        other_edges.append((edges[edge_idx2][0], edges[edge_idx1][0]))

            edges = [(edge[0], edge[1]) for edge in edges] + other_edges

        # duplicate P + Q for each answer
        offsets = []
        for kk, vv in enumerate(v['context']):
            offsets.append(orig_to_tok_map[kk] + len(question) + 1)
        # if k.startswith("docid_APW20000401.0150_sentid_10"):
        #     print(k)
        sample = {'offset': offsets,
                  'input_ids': tokenized_ids,
                  'mask_ids': mask_ids,
                  'segment_ids': segment_ids,
                  'question_id': k,
                    'q_head_mask' : bpe_to_node,
                    'question_len' : len(_alignment_q),
                    "edges": edges,
                    "q_event_word_idx" : [],
                    "q_tr_word_idx" : [],
                    "pos_labels": pos_labels,
                    "dep_labels": dep_labels,
                    "parser_tag_map" : parser_tag_map
                  }
        # add these three field for qualitative analysis
        if evaluation:
            sample['passage'] = v['context']
            sample['question'] = v['question']
            sample['question_cluster'] = v['question_cluster']
            sample['cluster_size'] = v['cluster_size']
        samples.append(sample)

        # check some example data
        if counter < 0:
            print(sample)
        counter += 1

    print("Maximum length after tokenization is: % s" % (max_len_global))
    return samples



def convert_to_features_roberta_qagnn(data, tokenizer, max_length=150, evaluation=False,
                                instance=True, end_to_end=False, suffix="", graph_construction=0, semi_struct_files = None, event_chain_files=None):
    pass
#   # each sample will have <s> Question </s> </s> Context </s>
#     samples = []
#     counter = 0
#     max_len_global = 0 # to show global max_len without truncating 

#     def get_edges(doc):
#         """get edges -> convert spacy edge index to original(nltk) index"""
#         # ('before', 'mark', 'SCONJ') # t.text, t.dep_, t.pos_
#         edges = [(token.i, token.head.i, int(token.pos_ == "VERB" or token.pos_=="NOUN"), token) for token in doc]
#         return edges 
#     if event_chain_files:
#         event_chains = json.load(open(event_chain_files[0]))
#         event_chains = dict(**event_chains, **json.load(open(event_chain_files[1])))
#         tmp_events = json.load(open("/root/temporal_reasoning/TimeBERT/code/ql/event_detector/context_to_event.json"))
#         def _get_event_chain(k):
#             try:
#                 event_chain = event_chains["_".join(k.split("_")[:-1])]
#             except KeyError: # when only one event, so no chain
#                 event_chain = tmp_events["_".join(k.split("_")[:-1])]
#                 assert sum(event_chain) == 1
#                 event_chain = [((("", event_chain.index(1)), ("", event_chain.index(1))), "VAGUE")]
#             return event_chain

#     if semi_struct_files:
#         semi_struct = json.load(open(semi_struct_files[0]))
#         semi_struct = dict(**semi_struct, **json.load(open(semi_struct_files[1])))       

#         def _get_semi_struct(k):
#             try:
#                 qa_pairs = semi_struct["_".join(k.split("_")[:-1])]
#             except KeyError:
#                 print("key error: ", "_".join(k.split("_")[:-1]))
#                 return [], []
#             qa_pairs = [aq[0] for aq in qa_pairs]
#             answers = [aq[0] for aq in qa_pairs]
#             questions = [aq[1] for aq in qa_pairs]
#             questions = tokenizer(questions, return_tensors='pt', padding='max_length', max_length=50)
#             return questions, answers

#     for k, v in tqdm(data.items(), desc="spacy & pre-process"):
#         #if counter==12:
#         #    print()

#         segment_ids = []
#         start_token = ['<s>']
#         #head_mask = [0]
#         question = tokenizer.tokenize(v['question'])     
        
#         #q_doc = nlp(v['question'])
#         #q_edges = get_edges(q_doc)
#         #_q_alignment = align_bpe_to_words(tokenizer, question, q_doc)
#         if semi_struct_files:
#             pseudo_questions, pseudo_answers = _get_semi_struct(k)
#         else:
#             pseudo_questions, pseudo_answers = [], []

#         new_tokens = ["</s>", "</s>"] # two sent sep symbols according the huggingface documentation
#         orig_to_tok_map = []
#         #head_mask += [0, 0]
#         for i, token in enumerate(v['context']):
#             orig_to_tok_map.append(len(new_tokens))
#             temp_tokens = tokenizer.tokenize(token)
#             new_tokens.extend(temp_tokens)
            
#         new_tokens.append("</s>")
#         length = len(new_tokens)
#         orig_to_tok_map.append(length)
#         assert len(orig_to_tok_map) == len(v['context']) + 1 # account for ending </s>

#         tokenized_ids = tokenizer.convert_tokens_to_ids(start_token + question + new_tokens)
#         if len(tokenized_ids) > max_len_global:
#             max_len_global = len(tokenized_ids)

#         if len(tokenized_ids) > max_length:
#             ending = tokenized_ids[-1]
#             tokenized_ids = tokenized_ids[:-(len(tokenized_ids) - max_length + 1)] + [ending]
            
#         segment_ids = [0] * len(tokenized_ids)
#         # mask ids                                                                                            
#         mask_ids = [1] * len(tokenized_ids)
        
#          # padding
#         if len(tokenized_ids) < max_length:
#             # Zero-pad up to the sequence length.
#             #padding = [tokenizer.pad_token_id] * (max_length - len(tokenized_ids))
#             zero_padding = [0] * (max_length - len(tokenized_ids))
#             tokenized_ids += zero_padding
#             mask_ids += zero_padding
#             segment_ids += zero_padding
#         assert len(tokenized_ids) == max_length


#         if graph_construction == 0: # syntactic graph, connect verb (for passages), connect v, n (for questions)

#             p_doc = nlp(" ".join(v['context']))
#             # for graph construction
#             edges = get_edges(p_doc)
#             # add node for question
#             if semi_struct_files:
#                 edges = [(elem[0]+1+len(pseudo_answers), elem[1]+1+len(pseudo_answers), elem[2], elem[3]) for elem in edges ]
#             else:
#                 edges = [(elem[0]+1, elem[1]+1, elem[2], elem[3]) for elem in edges]

#             # bpe와 spacy를 align 하겠다
#             _alignment = align_bpe_to_words(tokenizer, new_tokens[2:-1], p_doc)
#             # bpe 첫번째만 사용하겠다. 앞의 <s> question </s> </s> length 생각해서.
#             bpe_to_node = [align_idx[0] + len(question) + 3 for align_idx in _alignment]
#             assert len(bpe_to_node) == len(edges)

#             other_edges = []
#             """ 
#                 (head_index, tail_index, node_type_id, edge_type_id). 
#                 newly added edge type => 0
#                 question node type => 0
#             """
#             len_edges = len(edges)            
#             for edge_idx1 in range(len_edges):
#                 # connect to question
#                 # https://machinelearningknowledge.ai/tutorial-on-spacy-part-of-speech-pos-tagging/
#                 if edges[edge_idx1][3].text in v['question'] and edges[edge_idx1][3].pos_ in ["NOUN", "VERB", "PRON", "PROPN", "VERB", "ADJ"]:
#                     other_edges.append((0, edges[edge_idx1][0], 0, 0))
                
#                 for edge_idx2 in range(edge_idx1+1, len_edges):
#                     #if edge_idx1 == edge_idx2: continue                    
#                     # connect with sentences
#                     if edges[edge_idx1][3].dep_ == 'ROOT' and edges[edge_idx2][3].dep_ == 'ROOT':
#                         other_edges.append((edges[edge_idx1][0], edges[edge_idx2][0], 0, tagger_map[edges[edge_idx1][3].tag_]))
#                         other_edges.append((edges[edge_idx2][0], edges[edge_idx1][0], 0, tagger_map[edges[edge_idx2][3].tag_]))                
#                     elif edges[edge_idx1][3].pos_ == "VERB" and edges[edge_idx2][3].dep_ == "ROOT" and \
#                         find_root_token(edges[edge_idx1][3]) != find_root_token(edges[edge_idx2][3]) :
#                         other_edges.append((edges[edge_idx1][0], edges[edge_idx2][0], 1, tagger_map[edges[edge_idx1][3].tag_])) # 2 to mark newly added nodes
#                     elif edges[edge_idx2][3].pos_ == "VERB" and edges[edge_idx1][3].dep_ == "ROOT" and \
#                         find_root_token(edges[edge_idx1][3]) != find_root_token(edges[edge_idx2][3]) :
#                         other_edges.append((edges[edge_idx2][0], edges[edge_idx1][0], 1, tagger_map[edges[edge_idx2][3].tag_])) # 2 to mark newly added nodes

#             final_edges = [(edge[0], edge[1], depper_map[edge[3].dep_], tagger_map[edge[3].tag_]) for edge in edges] + \
#                 [(0,0,0,0)]+ \
#                     other_edges
                    
#         elif graph_construction == 1: # verb graph + syntactic question
#             p_doc = nlp(" ".join(v['context']))
#             q_doc = nlp(v['question'])
#             # for graph construction
#             edges = get_edges(p_doc)
#             q_edges = get_edges(q_doc)
#             # add node for question
#             edges = [(elem[0]+1+len(q_edges), elem[1]+1+len(q_edges), elem[2], elem[3]) for elem in edges]
#             edges = [(elem[0]+1, elem[1]+1, elem[2], elem[3]) for elem in q_edges] + edges

#             # bpe와 spacy를 align 하겠다
#             _alignment = align_bpe_to_words(tokenizer, new_tokens[2:-1], p_doc)
#             # bpe 첫번째만 사용하겠다. 앞의 <s> question </s> </s> length 생각해서.
#             bpe_to_node = [align_idx[0] + len(question) + 3 for align_idx in _alignment]
#             # question도 align
#             _alignment_q = align_bpe_to_words(tokenizer, question, q_doc)
#             bpe_to_node = [align_idx[0] + 1 for align_idx in _alignment_q] + bpe_to_node

#             assert len(bpe_to_node) == len(edges)

#             other_edges = []
#             """ 
#                 (head_index, tail_index, node_type_id, edge_type_id). 
#                 newly added edge type => 0
#                 question node type => 0
#             """
#             len_edges = len(edges)            
#             for edge_idx1 in range(len_edges):
#                 # connect to question
#                 # https://machinelearningknowledge.ai/tutorial-on-spacy-part-of-speech-pos-tagging/
#                 # for question, connect all (each question word to question node)
#                 if edge_idx1 < len(q_edges):
#                     other_edges.append((edges[edge_idx1][0], 0, 0, 0))
#                 # for passage, connect only nv
#                 elif edges[edge_idx1][3].text in v['question'] and edges[edge_idx1][3].pos_ in ["NOUN", "VERB", "PRON", "PROPN", "VERB", "ADJ"]:
#                     other_edges.append((0, edges[edge_idx1][0], 0, 0))
                
#                 for edge_idx2 in range(edge_idx1+1, len_edges):
#                     #if edge_idx1 == edge_idx2: continue                    
#                     # connect with sentences
#                     if edges[edge_idx1][3].dep_ == 'ROOT' and edges[edge_idx2][3].dep_ == 'ROOT':
#                         other_edges.append((edges[edge_idx1][0], edges[edge_idx2][0], 0, tagger_map[edges[edge_idx1][3].tag_]))
#                         other_edges.append((edges[edge_idx2][0], edges[edge_idx1][0], 0, tagger_map[edges[edge_idx2][3].tag_]))                
#                     elif edges[edge_idx1][3].pos_ == "VERB" and edges[edge_idx2][3].dep_ == "ROOT" and \
#                         find_root_token(edges[edge_idx1][3]) != find_root_token(edges[edge_idx2][3]) :
#                         other_edges.append((edges[edge_idx1][0], edges[edge_idx2][0], 1, tagger_map[edges[edge_idx1][3].tag_]))
#                     elif edges[edge_idx2][3].pos_ == "VERB" and edges[edge_idx1][3].dep_ == "ROOT" and \
#                         find_root_token(edges[edge_idx1][3]) != find_root_token(edges[edge_idx2][3]) :
#                         other_edges.append((edges[edge_idx2][0], edges[edge_idx1][0], 1, tagger_map[edges[edge_idx2][3].tag_]))
#             try:
#                 final_edges = [(edge[0], edge[1], depper_map[edge[3].dep_], tagger_map[edge[3].tag_]) for edge in edges] + \
#                     [(0,0,0,0)]+ \
#                         other_edges
#             except KeyError:
#                 exit("wrong label")

#         elif graph_construction == 2: 
#             #syntactic graph, question is treated as extended passage
#             event_pairs = _get_event_chain(k)
#             edges = []
#             event_indexs_e = []
#             all_event_indexs = []
#             # build passage question graph
#             for event_pair in event_pairs:
#                 if event_pair[1] == "BEFORE":
#                     edges.append((event_pair[0][0][1], event_pair[0][1][1], 1, 1))
#                     event_indexs_e.append(event_pair[0][1][1])
#                 elif event_pair[1] == "AFTER":
#                     edges.append((event_pair[0][1][1], event_pair[0][0][1], 1, 1))
#                     event_indexs_e.append(event_pair[0][0][1])
#                 # for start of chain
#                 all_event_indexs.append(event_pair[0][0][1])
#                 all_event_indexs.append(event_pair[0][1][1])
#             # 각 pair 들을 0, 1, ... 로 매핑시켜줘야함...
#             all_event_indexs = list(set(all_event_indexs))
#             all_event_indexs.sort()
#             index_to_node_map = {all_event_index: i + 1 for i, all_event_index in enumerate(all_event_indexs)} # +1 for q event
#             edges = [(index_to_node_map[edge[0]], index_to_node_map[edge[1]], edge[2], edge[3]) for edge in edges]
#             edges = list(set(edges))
#             #edges = list(set(edges))
#             # connect question for start of the chain
#             q_to_event_nodes = list(set(all_event_indexs) - set(event_indexs_e))
#             other_edges = [(0, index_to_node_map[event_node], 0, 0) for event_node in q_to_event_nodes]
#             final_edges = edges + [(0,0,0,0)] + other_edges
            
#             # here, bpe_to_node = map event to bpe
#             bpe_to_node = [orig_to_tok_map[event_index] + len(question) + 1 for event_index in all_event_indexs]
            
#         elif graph_construction == 3: 
#             #question is treated as extended passage. question = syntactic, passage = semantic.
#             # ignore q node
#             # p - q: same mention. check by : is q_lemma in p?
#             # q - p: start node
#             # node type: 0 = question node 1 = question node 2 = passage node
#             # edge type: #dep + 1 + (question avg node self edge)  + 1 (p-q) + 1(q-p) + 1 (p inner)
#             q_doc = nlp(v['question'])
#             q_edges = get_edges(q_doc)
#             _alignment_q = align_bpe_to_words(tokenizer, question, q_doc)
#             _bpe_to_node_q = [align_idx[0] + 1 for align_idx in _alignment_q]

#             event_pairs = _get_event_chain(k)
#             p_edges = []
#             event_indexs_e = []
#             all_event_indexs = []
#             for event_pair in event_pairs:
#                 if event_pair[1] == "BEFORE":
#                     p_edges.append((event_pair[0][0][1], event_pair[0][1][1], 3, 2))
#                     event_indexs_e.append(event_pair[0][1][1])
#                 elif event_pair[1] == "AFTER":
#                     p_edges.append((event_pair[0][1][1], event_pair[0][0][1], 3, 2))
#                     event_indexs_e.append(event_pair[0][0][1])
#                 # for start of chain
#                 all_event_indexs.append(tuple(event_pair[0][0]))
#                 all_event_indexs.append(tuple(event_pair[0][1]))
#             p_edges = list(set(p_edges))
#             # 각 pair 들을 0, 1, ... 로 매핑시켜줘야함..
#             all_event_indexs = list(set(all_event_indexs))
#             all_event_indexs.sort(key=lambda x: x[1])
#             index_to_node_map = {all_event_index[1]: i + len(q_edges) + 1 for i, all_event_index in enumerate(all_event_indexs)}
            
#             p_edges = [(index_to_node_map[edge[0]], index_to_node_map[edge[1]], edge[2], edge[3]) for edge in p_edges]

#             # find start event idx
#             # 전체 event에서 
#             q_to_event_nodes = list(set([event_index[1] for event_index in all_event_indexs]) - set(event_indexs_e)) 
            
#             other_edges = []
#             for edge in q_edges:
#                 if edge[3].dep_ == "ROOT": 
#                     # additional edges 1. question ROOT - start of the chain
#                     q_root_node = edge[0]
#                 # additional edges 2. question node - same mention
#                 for each_event in all_event_indexs:
#                     if (edge[3].lemma_ in each_event[0]) or (edge[3].text in each_event[0]):
#                         other_edges.append((index_to_node_map[each_event[1]], edge[0], 1, 2)) # p to q : 1, # p node 2
#                         other_edges.append((edge[0], index_to_node_map[each_event[1]], 2, 1)) # q to p : 2 # q node 1
#             # additional edges 1. question ROOT - start of the chain
#             other_edges += [(q_root_node, index_to_node_map[event_node], 1, 1) for event_node in q_to_event_nodes]       
            
#             # get final edge
#             q_edges = [(edge[0]+1, edge[1]+1, depper_map[edge[3].dep_] + 2, 1) for edge in q_edges] # depper labels + 3 other labels
#             final_edges = q_edges + p_edges + [(0,0,0,0)] + other_edges
            
#             # here, bpe_to_node = map event to bpe
#             bpe_to_node = _bpe_to_node_q + [orig_to_tok_map[event_index[1]] + len(question) + 1 for event_index in all_event_indexs]

#         elif graph_construction == 4: # p edges -> remove some POS
#             p_doc = nlp(" ".join(v['context']))
#             q_doc = nlp(v['question'])
#             # for graph construction
#             edges = get_edges(p_doc)
#             q_edges = get_edges(q_doc)
#             new_p_edges = []
#             for p_edge in edges:
#                 if p_edge[3].pos_ in ["NOUN", "VERB", "PRON", "PROPN", "ADJ", "AUX", "NUM"] or p_edge[3].dep_ == "ROOT":
#                     tail = p_edge[3].head
#                     while tail.pos_ not in ["NOUN", "VERB", "PRON", "PROPN", "ADJ", "AUX", "NUM"] and tail.dep_ != "ROOT" and tail.i != tail.head.i:
#                         tail = tail.head
#                     new_p_edges.append((p_edge[3].i, tail.i, p_edge[2], p_edge[3]))

#             # add node for question
#             edges = [(elem[0], elem[1], elem[2], elem[3]) for elem in new_p_edges]

#             # bpe와 spacy를 align 하겠다
#             _alignment = align_bpe_to_words(tokenizer, new_tokens[2:-1], p_doc)
#             # bpe 첫번째만 사용하겠다. 앞의 <s> question </s> </s> length 생각해서.
#             bpe_to_node = [align_idx[0] + len(question) + 3 for align_idx in _alignment]
#             # only selected bpes
#             _edge_to_include = list(set([new_p_edge[0] for new_p_edge in new_p_edges] + [new_p_edge[1] for new_p_edge in new_p_edges]))
#             _edge_to_include.sort()
#             bpe_to_node = [align_idx for i, align_idx in enumerate(bpe_to_node) if i in _edge_to_include]
#             _edge_to_include = {edge_idx : i for i, edge_idx in enumerate(_edge_to_include)}
#             edges = [(_edge_to_include[elem[0]] + 1 + len(q_edges), _edge_to_include[elem[1]] + 1 + len(q_edges) , elem[2], elem[3]) for elem in new_p_edges]

#             # question도 align
#             edges = [(elem[0]+1, elem[1]+1, elem[2], elem[3]) for elem in q_edges] + edges
#             _alignment_q = align_bpe_to_words(tokenizer, question, q_doc)
#             bpe_to_node = [align_idx[0] + 1 for align_idx in _alignment_q] + bpe_to_node

#             assert len(bpe_to_node) == len(edges)

#             other_edges = []
#             """ 
#                 (head_index, tail_index, edge_type_id, node_type_id). 
#                 # edge to question avg node = 0
#                 # edge to question - passage = 1
#                 # edge to passage - question = 2
#                 # edge between passage = 3
#                 passage node type = 2
#                 question node type = 1
#                 question avg node type => 0

#             """
#             len_edges = len(edges)            
#             for edge_idx1 in range(len_edges):
#                 # connect to question
#                 # https://machinelearningknowledge.ai/tutorial-on-spacy-part-of-speech-pos-tagging/
#                 # for question, connect all (each question word to question node)
#                 if edge_idx1 < len(q_edges):
#                     other_edges.append((edges[edge_idx1][0], 0, 0, 0))
#                     for edge_idx2 in range(len(q_edges), len_edges):
#                         #if edge_idx1 == edge_idx2: continue                    
#                         # connect q - p. same mention or ROOT
#                         if (edges[edge_idx1][3].dep_ == 'ROOT' and edges[edge_idx2][3].dep_ == 'ROOT') or \
#                          edges[edge_idx1][3].lemma_.lower() == edges[edge_idx2][3].lemma_.lower():
#                             other_edges.append((edges[edge_idx1][0], edges[edge_idx2][0], 1, 1))
#                             other_edges.append((edges[edge_idx2][0], edges[edge_idx1][0], 2, 2))
#                 else:
#                     for edge_idx2 in range(edge_idx1+1, len_edges):
#                         #if edge_idx1 == edge_idx2: continue                    
#                         # connect within passage. 
#                         if edges[edge_idx1][3].dep_ == 'ROOT' and edges[edge_idx2][3].dep_ == 'ROOT':
#                             other_edges.append((edges[edge_idx1][0], edges[edge_idx2][0], 3, 2))
#                             other_edges.append((edges[edge_idx2][0], edges[edge_idx1][0], 3, 2))                
#                         elif edges[edge_idx1][3].pos_ == "VERB" and edges[edge_idx2][3].dep_ == "ROOT" and \
#                             find_root_token(edges[edge_idx1][3]) != find_root_token(edges[edge_idx2][3]) :
#                             other_edges.append((edges[edge_idx1][0], edges[edge_idx2][0], 3, 2)) 
#                         elif edges[edge_idx2][3].pos_ == "VERB" and edges[edge_idx1][3].dep_ == "ROOT" and \
#                             find_root_token(edges[edge_idx1][3]) != find_root_token(edges[edge_idx2][3]) :
#                             other_edges.append((edges[edge_idx2][0], edges[edge_idx1][0], 3, 2))
#             try:
#                 final_edges = [(edge[0], edge[1], depper_map[edge[3].dep_], 1) for edge in edges[:len(q_edges)]] + \
#                 [(edge[0], edge[1], 3, 2) for edge in edges[len(q_edges):]] + \
#                     [(0,0,0,0)]+ \
#                         other_edges
#             except KeyError:
#                 exit("wrong label")


#         # for pseudo questions
#         if pseudo_answers:
#             pseudo_edges = []
#             for idx, answers in enumerate(pseudo_answers):
#                 answers = answers.split(",")
#                 for edge in edges:
#                     if edge[3].text in answers:
#                         pseudo_edges.append((idx+1, edge[0], 0, 0))
#             final_edges += pseudo_edges

#         # pad
#         #if len(head_mask) < max_length:
#         #    head_mask += [0] * (max_length - len(head_mask))

#         # construct a sample
#         # offset: </s> </s> counted in orig_to_tok_map already, so only need to worry about <s>
#         if end_to_end:
#             # duplicate P + Q for each answer
#             labels, offsets = [], []
#             for kk, vv in enumerate(v['answers']['labels']):
#                 labels.append(vv)
#                 offsets.append(orig_to_tok_map[kk] + len(question) + 1)
#             # offsets == (idx of head_masks == 1 )[question len:]
#             sample = {'label': labels,
#                       #'events': events,
#                       'offset': offsets,
#                       'input_ids': tokenized_ids,
#                       'mask_ids': mask_ids,
#                       'segment_ids': segment_ids,
#                       'question_id': k,
#                       #'q_head_mask' : head_mask,
#                       #'question_len' : question_len,
#                       "edges": final_edges,
#                       "bpe_to_node": bpe_to_node,
#                       "extra_info": pseudo_questions
#                       }
#             # add these three field for qualitative analysis                                            
#             if evaluation:
#                 sample['passage'] = v['context']
#                 sample['question'] = v['question']
#                 sample['question_cluster'] = v['question_cluster']
#                 sample['cluster_size'] = v['cluster_size']
#                 sample['answer'] = v['answers']
#                 sample['individual_answers'] = [a['labels'] for a in v['individual_answers']]
            
#             samples.append(sample)
#         else:
#             # no duplicate P + Q
#             labels, offsets = [], []
#             for vv in v['answers'].values():
#                 labels.append(vv['label'])
#                 offsets.append(orig_to_tok_map[vv['idx']] + len(question) + 1)
#             assert len(offsets) == len(edges)
#             sample = {'label': labels,
#                       'offset': offsets,
#                       'input_ids': tokenized_ids,
#                       'mask_ids': mask_ids,
#                       'segment_ids': segment_ids,
#                       'question_id': k,
#                       "edges": final_edges,
#                       "bpe_to_node": bpe_to_node,
#                       "extra_info": pseudo_questions
#                       }
            
#             # add these three field for qualitative analysis         
#             if evaluation:
#                 sample['passage'] = v['context']
#                 sample['question'] = v['question']
#                 sample['answer'] = v['answers']
#                 sample['question_cluster'] = v['question_cluster']
#                 sample['cluster_size'] = v['cluster_size']
#                 individual_answers = []
#                 for vv in v['individual_answers']:
#                     individual_answers.append([a['label'] for a in vv.values()])
#                 sample['individual_answers'] = individual_answers
                
#             samples.append(sample)
            
#         # check some example data
#         if counter < 0:
#             print(sample)
#         counter += 1

#     print("Maximum length after tokenization is: % s" % (max_len_global))

    # if not evaluation:
    #    pickle.dump(samples, open(pickle_file_name, 'wb'))

 #   return samples

def convert_to_features_roberta_graph2(data, tokenizer, max_length=150, max_question_length = 0, 
                                       evaluation=False, instance=True, end_to_end=False, 
                                       is_qdgat=0, suffix="", graph_construction = 0, event_chain_files=None,
                                       mask_what=False, train_ratio = 1.0):
    add_space = True
    print(f"train_roberta_{graph_construction}_{mask_what}_{add_space}_{train_ratio}.pkl")
    if not evaluation and os.path.exists(f"train_roberta_{graph_construction}_{mask_what}_{add_space}_{train_ratio}.pkl"):
        print("load data from pickle. If you revised data reader, delete this script")
        samples =pickle.load(open(f"train_roberta_{graph_construction}_{mask_what}_{add_space}_{train_ratio}.pkl", 'rb'))
        return samples

    # each sample will have <s> Question </s> </s> Context </s>
    samples = []
    counter = 0
    max_len_global = 0 # to show global max_len without truncating 

    def get_edges(doc):
        if graph_construction == 0:
            edges = [(token.i, token.head.i, int(token.dep_=="ROOT"), token) for token in doc]
        elif graph_construction == 1:
            edges = [(token.i, token.head.i, int(token.dep_=="ROOT" or token.pos_=="VERB"), token) for token in doc]
        else: 
            edges = [(token.i, token.head.i, int(token.dep_=="ROOT"), token) for token in doc]
        return edges

    if event_chain_files:
        event_chains = json.load(open(event_chain_files[0]))
        event_chains = dict(**event_chains, **json.load(open(event_chain_files[1])))
        tmp_events = json.load(open("/root/temporal_reasoning/TimeBERT/code/ql/event_detector/context_to_event.json"))
        def _get_event_chain(k):
            try:
                event_chain = event_chains["_".join(k.split("_")[:-1])]
            except KeyError: # when only one event, so no chain
                event_chain = tmp_events["_".join(k.split("_")[:-1])]
                assert sum(event_chain) == 1
                event_chain = [((("", event_chain.index(1)), ("", event_chain.index(1))), "VAGUE")]
            return event_chain

    for k, v in tqdm(data.items(), desc="spacy & pre-process"):
        #if counter==12:
        #    print()
        if train_ratio < 1 and counter > len(data) * train_ratio:
            continue
        segment_ids = []
        start_token = ['<s>']
        question = tokenizer.tokenize(v['question'])
        q_doc = nlp(v['question'])
        q_edges = get_edges(q_doc)
        # get question len
        
        new_tokens = ["</s>", "</s>"] # two sent sep symbols according the huggingface documentation
        orig_to_tok_map = []
        for i, token in enumerate(v['context']):
            orig_to_tok_map.append(len(new_tokens))
            if add_space:
                if i == 0 or any(s in token for s in string.punctuation):
                    temp_tokens = tokenizer.tokenize(token)
                else:
                    temp_tokens = tokenizer.tokenize(" " + token)
            else:
                temp_tokens = tokenizer.tokenize(token)
            new_tokens.extend(temp_tokens)
            
        new_tokens.append("</s>")
        length = len(new_tokens)
        orig_to_tok_map.append(length)
        assert len(orig_to_tok_map) == len(v['context']) + 1 # account for ending </s>

        # if mask_what:
        #     # change 'what'
        #     new_question = []
        #     changed_flag = False
        #     for q_word in question:
        #         # change the first occurnce
        #         if not changed_flag and q_word in ["What", "what", "Ġwhat"]:
        #             changed_flag = True
        #             new_question.append(tokenizer.mask_token)  
        #         else:          
        #             new_question.append(q_word)
        #     if changed_flag and new_question[-1] == "?":
        #         new_question[-1] == "."
        #     tokenized_ids = tokenizer.convert_tokens_to_ids(start_token + new_question + new_tokens) 
        # else:

        tokenized_ids = tokenizer.convert_tokens_to_ids(start_token + question + new_tokens)
        if len(tokenized_ids) > max_len_global:
            max_len_global = len(tokenized_ids)

        if len(tokenized_ids) > max_length:
            ending = tokenized_ids[-1]
            tokenized_ids = tokenized_ids[:-(len(tokenized_ids) - max_length + 1)] + [ending]
            
        segment_ids = [0] * len(tokenized_ids)
        # mask ids                                                                                            
        mask_ids = [1] * len(tokenized_ids)
        
         # padding
        if len(tokenized_ids) < max_length:
            # Zero-pad up to the sequence length.
            #padding = [tokenizer.pad_token_id] * (max_length - len(tokenized_ids))
            zero_padding = [0] * (max_length - len(tokenized_ids))
            tokenized_ids += zero_padding
            mask_ids += zero_padding
            segment_ids += zero_padding
        assert len(tokenized_ids) == max_length

        if graph_construction == 0 or graph_construction == 1:
            p_doc = nlp(" ".join(v['context']))
            # for graph construction
            p_edges = get_edges(p_doc)

            #p_edges = [(elem[0], elem[1], elem[2], elem[3]) for elem in p_edges]
            q_edges = [(elem[0], elem[1], elem[2]) for elem in q_edges] 

            # bpe와 spacy를 align 하겠다
            _alignment = align_bpe_to_words(tokenizer, new_tokens[2:-1], p_doc)
            # bpe 첫번째만 사용하겠다. 앞의 <s> question </s> </s> length 생각해서.
            bpe_to_node = [align_idx[0] + len(question) + 3 for align_idx in _alignment]
            # question도 align
            _alignment_q = align_bpe_to_words(tokenizer, question, q_doc)
            bpe_to_node = [align_idx[0] + 1 for align_idx in _alignment_q] + bpe_to_node

            pos_labels = [tagger_map[token.tag_] for token in p_doc]
            dep_labels = [depper_map[token.dep_] for token in p_doc]
            parser_tag_map = [p_edge[0] for p_edge in p_edges]

            # add question length for numgnn
            p_edges = [(elem[0]+max_question_length, elem[1]+max_question_length, elem[2]) for elem in p_edges] 

            edges = q_edges + p_edges
            assert len(bpe_to_node) == len(edges)
            
            verb_edges = []
            len_edges = len(edges)
            for edge_idx1 in range(len_edges):
                for edge_idx2 in range(edge_idx1+1, len_edges):
                    if edges[edge_idx1][2] == 1 and edges[edge_idx2][2] == 1:
                        verb_edges.append((edges[edge_idx1][0], edges[edge_idx2][0], 1))  
                        verb_edges.append((edges[edge_idx2][0], edges[edge_idx1][0], 1))            

            edges += verb_edges
        elif graph_construction == 2: # same mention
            p_doc = nlp(" ".join(v['context']))
            # for graph construction
            p_edges = get_edges(p_doc)

            # bpe와 spacy를 align 하겠다
            _alignment = align_bpe_to_words(tokenizer, new_tokens[2:-1], p_doc)
            # bpe 첫번째만 사용하겠다. 앞의 <s> question </s> </s> length 생각해서.
            bpe_to_node = [align_idx[0] + len(question) + 3 for align_idx in _alignment]
            # question도 align
            _alignment_q = align_bpe_to_words(tokenizer, question, q_doc)
            bpe_to_node = [align_idx[0] + 1 for align_idx in _alignment_q] + bpe_to_node

            pos_labels = [tagger_map[token.tag_] for token in p_doc]
            dep_labels = [depper_map[token.dep_] for token in p_doc]
            parser_tag_map = [p_edge[0] for p_edge in p_edges]

            # add question length for numgnn
            p_edges = [(elem[0]+max_question_length, elem[1]+max_question_length, elem[2], elem[3]) for elem in p_edges] 

            edges = q_edges + p_edges
            assert len(bpe_to_node) == len(edges)
            
            # 지금 문제: 다른 sentence에 있는 건 못본다?

            other_edges = []
            for edge_idx1 in range(len(edges)):
                # https://machinelearningknowledge.ai/tutorial-on-spacy-part-of-speech-pos-tagging/                
                for edge_idx2 in range(edge_idx1+1, len(edges)):
                    #if edge_idx1 == edge_idx2: continue                    
                    # connect with sentences
                    if edges[edge_idx1][3].dep_ == 'ROOT' and edges[edge_idx2][3].dep_ == 'ROOT':
                        other_edges.append((edges[edge_idx1][0], edges[edge_idx2][0]))
                        other_edges.append((edges[edge_idx2][0], edges[edge_idx1][0]))                
                    if edges[edge_idx1][3].lemma_ == edges[edge_idx2][3].lemma_ and \
                        edges[edge_idx1][3].pos_ in ["NOUN", "VERB", "PRON", "PROPN", "ADJ", "AUX", "NUM"] :
                        other_edges.append((edges[edge_idx1][0], edges[edge_idx2][0]))
                        other_edges.append((edges[edge_idx2][0], edges[edge_idx1][0]))

            edges = [(edge[0], edge[1]) for edge in edges] + other_edges
        
        elif graph_construction == 3: # syntactic q + semantic p            
            _alignment_q = align_bpe_to_words(tokenizer, question, q_doc)
            _bpe_to_node_q = [align_idx[0] + 1 for align_idx in _alignment_q]

            pos_labels = []
            dep_labels = []
            parser_tag_map = []

            event_pairs = _get_event_chain(k)
            p_edges = []
            event_indexs_e = []
            all_event_indexs = []
            for event_pair in event_pairs:
                if event_pair[1] == "BEFORE":
                    p_edges.append((event_pair[0][0][1], event_pair[0][1][1]))
                    event_indexs_e.append(event_pair[0][1][1])
                elif event_pair[1] == "AFTER":
                    p_edges.append((event_pair[0][1][1], event_pair[0][0][1]))
                    event_indexs_e.append(event_pair[0][0][1])
                # for start of chain
                all_event_indexs.append(tuple(event_pair[0][0]))
                all_event_indexs.append(tuple(event_pair[0][1]))
            p_edges = list(set(p_edges))
            # 각 pair 들을 0, 1, ... 로 매핑시켜줘야함..
            all_event_indexs = list(set(all_event_indexs))
            all_event_indexs.sort(key=lambda x: x[1])
            index_to_node_map = {all_event_index[1]: i + max_question_length for i, all_event_index in enumerate(all_event_indexs)}
            
            p_edges = [(index_to_node_map[edge[0]], index_to_node_map[edge[1]]) for edge in p_edges]

            # find start event idx
            # 전체 event에서 
            q_to_event_nodes = list(set([event_index[1] for event_index in all_event_indexs]) - set(event_indexs_e)) 
            
            other_edges = []
            for edge in q_edges:
                if edge[3].dep_ == "ROOT": 
                    # additional edges 1. question ROOT - start of the chain
                    q_root_node = edge[0]
                # additional edges 2. question node - same mention
                for each_event in all_event_indexs:
                    if (edge[3].lemma_ in each_event[0]) or (edge[3].text in each_event[0]):
                        other_edges.append((index_to_node_map[each_event[1]], edge[0])) # p to q : 1, # p node 2
                        other_edges.append((edge[0], index_to_node_map[each_event[1]])) 
            # additional edges 1. question ROOT - start of the chain
            other_edges += [(q_root_node, index_to_node_map[event_node]) for event_node in q_to_event_nodes]       
            
            # get final edge
            q_edges = [(edge[0], edge[1]) for edge in q_edges] 
            edges = q_edges + p_edges + other_edges
            
            # here, bpe_to_node = map event to bpe
            bpe_to_node = _bpe_to_node_q + [orig_to_tok_map[event_index[1]] + len(question) + 1 for event_index in all_event_indexs]


        q_event_word_idx, q_tr_word_idx = [], []

        # construct a sample
        # offset: </s> </s> counted in orig_to_tok_map already, so only need to worry about <s>
        if end_to_end:
            # duplicate P + Q for each answer
            labels, offsets = [], []
            events =[]
            for kk, vv in enumerate(v['answers']['labels']):
                labels.append(vv)
                offsets.append(orig_to_tok_map[kk] + len(question) + 1)
            for vv in v['answers']['types']:
                events.append(vv)
            # offsets == (idx of head_masks == 1 )[question len:]
            sample = {'label': labels,
                      'events': events,
                      'offset': offsets,
                      'input_ids': tokenized_ids,
                      'mask_ids': mask_ids,
                      'segment_ids': segment_ids,
                      'question_id': k,
                      'q_head_mask' : bpe_to_node,
                      'question_len' : len(_alignment_q),
                      "edges": edges,
                      "q_event_word_idx" : q_event_word_idx,
                      "q_tr_word_idx" : q_tr_word_idx,
                      "pos_labels": pos_labels,
                      "dep_labels": dep_labels,
                      "parser_tag_map" : parser_tag_map
                      }
            # add these three field for qualitative analysis                                            
            if evaluation:
                sample['passage'] = v['context']
                sample['question'] = v['question']
                sample['question_cluster'] = v['question_cluster']
                sample['cluster_size'] = v['cluster_size']
                sample['answer'] = v['answers']
                sample['individual_answers'] = [a['labels'] for a in v['individual_answers']]
            
            samples.append(sample)
        else:
            # no duplicate P + Q
            labels, offsets = [], []
            events = []
            for vv in v['answers'].values():
                labels.append(vv['label'])
                offsets.append(orig_to_tok_map[vv['idx']] + len(question) + 1)
            for vv in v['answers']['types']:
                events.append(vv)
            sample = {'label': labels,
                      'events': events,
                      'offset': offsets,
                      'input_ids': tokenized_ids,
                      'mask_ids': mask_ids,
                      'segment_ids': segment_ids,
                      'question_id': k,
                      'q_head_mask' : bpe_to_node,
                      'question_len' : len(q_edges),
                      "edges": edges,
                      "q_event_word_idx" : q_event_word_idx,
                      "q_tr_word_idx" : q_tr_word_idx,
                      "pos_labels": pos_labels,
                      "dep_labels": dep_labels,
                      "parser_tag_map" : parser_tag_map
                      }
            
            # add these three field for qualitative analysis         
            if evaluation:
                sample['passage'] = v['context']
                sample['question'] = v['question']
                sample['answer'] = v['answers']
                sample['question_cluster'] = v['question_cluster']
                sample['cluster_size'] = v['cluster_size']
                individual_answers = []
                for vv in v['individual_answers']:
                    individual_answers.append([a['label'] for a in vv.values()])
                sample['individual_answers'] = individual_answers
                
            samples.append(sample)
            
        # check some example data
        if counter < 0:
            print(sample)
        counter += 1
    print("Maximum length after tokenization is: % s" % (max_len_global))

    if not evaluation:
        pickle.dump(samples, open(f"train_roberta_{graph_construction}_{mask_what}_{add_space}_{train_ratio}.pkl", 'wb'))
    return samples

def convert_to_features_roberta_graph2_no_label(data, tokenizer, max_length=150, max_question_length=35, evaluation=False, graph_construction = 0):
    # each sample will have <s> Question </s> </s> Context </s>
    samples = []
    counter = 0
    max_len_global = 0  # to show global max_len without truncating
    
    def get_edges(doc):
        if graph_construction == 0:
            edges = [(token.i, token.head.i, int(token.dep_=="ROOT"), token) for token in doc]
        elif graph_construction == 1:
            edges = [(token.i, token.head.i, int(token.dep_=="ROOT" or token.pos_=="VERB"), token) for token in doc]
        else: 
            edges = [(token.i, token.head.i, int(token.dep_=="ROOT"), token) for token in doc]
        return edges

    for k, v in tqdm(data.items()):
        counter += 1
        
        start_token = ['<s>']
        question = tokenizer.tokenize(v['question'])
        q_doc = nlp(v['question'])
        q_edges = get_edges(q_doc)
        
        new_tokens = ["</s>", "</s>"]  # two sent sep symbols according the huggingface documentation
        orig_to_tok_map = []
        for i, token in enumerate(v['context']):
            orig_to_tok_map.append(len(new_tokens))
            if i == 0 or any(s in token for s in string.punctuation):
                temp_tokens = tokenizer.tokenize(token)
            else:
                temp_tokens = tokenizer.tokenize(" " + token)
            #temp_tokens = tokenizer.tokenize(token)
            new_tokens.extend(temp_tokens)

        new_tokens.append("</s>")
        length = len(new_tokens)
        orig_to_tok_map.append(length)
        assert len(orig_to_tok_map) == len(v['context']) + 1  # account for ending </s>

        tokenized_ids = tokenizer.convert_tokens_to_ids(start_token + question + new_tokens)
        if len(tokenized_ids) > max_len_global:
            max_len_global = len(tokenized_ids)

        if len(tokenized_ids) > max_length:
            ending = tokenized_ids[-1]
            tokenized_ids = tokenized_ids[:-(len(tokenized_ids) - max_length + 1)] + [ending]

        segment_ids = [0] * len(tokenized_ids)
        # mask ids
        mask_ids = [1] * len(tokenized_ids)

        # padding
        if len(tokenized_ids) < max_length:
            # Zero-pad up to the sequence length.
            padding = [0] * (max_length - len(tokenized_ids))
            tokenized_ids += padding
            mask_ids += padding
            segment_ids += padding
        assert len(tokenized_ids) == max_length


        if graph_construction == 2: # same mention
            p_doc = nlp(" ".join(v['context']))
            # for graph construction
            p_edges = get_edges(p_doc)

            # bpe와 spacy를 align 하겠다
            _alignment = align_bpe_to_words(tokenizer, new_tokens[2:-1], p_doc)
            # bpe 첫번째만 사용하겠다. 앞의 <s> question </s> </s> length 생각해서.
            bpe_to_node = [align_idx[0] + len(question) + 3 for align_idx in _alignment]
            # question도 align
            _alignment_q = align_bpe_to_words(tokenizer, question, q_doc)
            bpe_to_node = [align_idx[0] + 1 for align_idx in _alignment_q] + bpe_to_node

            pos_labels = [tagger_map[token.tag_] for token in p_doc]
            dep_labels = [depper_map[token.dep_] for token in p_doc]
            parser_tag_map = [p_edge[0] for p_edge in p_edges]

            # add question length for numgnn
            p_edges = [(elem[0]+max_question_length, elem[1]+max_question_length, elem[2], elem[3]) for elem in p_edges] 

            edges = q_edges + p_edges
            assert len(bpe_to_node) == len(edges)
            
            other_edges = []
            for edge_idx1 in range(len(edges)):
                # https://machinelearningknowledge.ai/tutorial-on-spacy-part-of-speech-pos-tagging/                
                for edge_idx2 in range(edge_idx1+1, len(edges)):
                    #if edge_idx1 == edge_idx2: continue                    
                    # connect with sentences
                    if edges[edge_idx1][3].dep_ == 'ROOT' and edges[edge_idx2][3].dep_ == 'ROOT':
                        other_edges.append((edges[edge_idx1][0], edges[edge_idx2][0]))
                        other_edges.append((edges[edge_idx2][0], edges[edge_idx1][0]))                
                    if edges[edge_idx1][3].lemma_ == edges[edge_idx2][3].lemma_ and \
                        edges[edge_idx1][3].pos_ in ["NOUN", "VERB", "PRON", "PROPN", "ADJ", "AUX", "NUM"] :
                        other_edges.append((edges[edge_idx1][0], edges[edge_idx2][0]))
                        other_edges.append((edges[edge_idx2][0], edges[edge_idx1][0]))

            edges = [(edge[0], edge[1]) for edge in edges] + other_edges

        q_event_word_idx, q_tr_word_idx = [], []
        # construct a sample
        # offset: </s> </s> counted in orig_to_tok_map already, so only need to worry about <s>

        # duplicate P + Q for each answer
        offsets = []
        for kk, vv in enumerate(v['context']):
            offsets.append(orig_to_tok_map[kk] + len(question) + 1)
        # if k.startswith("docid_APW20000401.0150_sentid_10"):
        #     print(k)
        sample = {'offset': offsets,
                  'input_ids': tokenized_ids,
                  'mask_ids': mask_ids,
                  'segment_ids': segment_ids,
                  'question_id': k,
                    'q_head_mask' : bpe_to_node,
                    'question_len' : len(_alignment_q),
                    "edges": edges,
                    "q_event_word_idx" : q_event_word_idx,
                    "q_tr_word_idx" : q_tr_word_idx,
                    "pos_labels": pos_labels,
                    "dep_labels": dep_labels,
                    "parser_tag_map" : parser_tag_map
                  }
        # add these three field for qualitative analysis
        if evaluation:
            sample['passage'] = v['context']
            sample['question'] = v['question']
            sample['question_cluster'] = v['question_cluster']
            sample['cluster_size'] = v['cluster_size']
        samples.append(sample)

        # check some example data
        if counter < 0:
            print(sample)
        

    print("Maximum length after tokenization is: % s" % (max_len_global))
    return samples


def convert_to_features_deberta_graph2(data, tokenizer, max_length=150, max_question_length = 0, 
                                       evaluation=False, instance=True, end_to_end=False, 
                                       is_qdgat=0, suffix="", graph_construction = 0, event_chain_files=None,
                                       mask_what=False, train_ratio = 1.0):
    add_space = False
    print(f"train_{tokenizer.__class__.__name__}_{graph_construction}_{mask_what}_{add_space}_{max_length}_{max_question_length}.pkl")
    if not evaluation and os.path.exists(f"train_{tokenizer.__class__.__name__}_{graph_construction}_{mask_what}_{add_space}_{max_length}_{max_question_length}.pkl"):
        print("load data from pickle. If you revised data reader, delete this script")
        samples =pickle.load(open(f"train_{tokenizer.__class__.__name__}_{graph_construction}_{mask_what}_{add_space}_{max_length}_{max_question_length}.pkl", 'rb'))
        return samples

    # each sample will have <s> Question </s> </s> Context </s>
    samples = []
    counter = 0
    max_len_global = 0 # to show global max_len without truncating 

    def get_edges(doc):
        if graph_construction == 0:
            edges = [(token.i, token.head.i, int(token.dep_=="ROOT"), token) for token in doc]
        elif graph_construction == 1:
            edges = [(token.i, token.head.i, int(token.dep_=="ROOT" or token.pos_=="VERB"), token) for token in doc]
        else: 
            edges = [(token.i, token.head.i, int(token.dep_=="ROOT"), token) for token in doc]
        return edges

    for k, v in tqdm(data.items(), desc="spacy & pre-process"):
        #if counter==12:
        #    print()
        if train_ratio < 1 and counter > len(data) * train_ratio:
            continue
        segment_ids = []
        start_token = ['[CLS]']
        v['question'] = v['question'].strip()
        question = tokenizer.tokenize(v['question'])
        q_doc = nlp(v['question'])
        q_edges = get_edges(q_doc)
        # get question len
        
        new_tokens = ["[SEP]"] # two sent sep symbols according the huggingface documentation
        orig_to_tok_map = []
        for i, token in enumerate(v['context']):
            orig_to_tok_map.append(len(new_tokens))
            if add_space:
                if i == 0 or any(s in token for s in string.punctuation):
                    temp_tokens = tokenizer.tokenize(token)
                else:
                    temp_tokens = tokenizer.tokenize(" " + token)
            else:
                temp_tokens = tokenizer.tokenize(token)
            new_tokens.extend(temp_tokens)
            
        new_tokens.append("[SEP]")
        length = len(new_tokens)
        orig_to_tok_map.append(length)
        assert len(orig_to_tok_map) == len(v['context']) + 1 # account for ending </s>
 
        tokenized_ids = tokenizer.convert_tokens_to_ids(start_token + question + new_tokens)
        if len(tokenized_ids) > max_len_global:
            max_len_global = len(tokenized_ids)

        if len(tokenized_ids) > max_length:
            logger.info("truncate", v['context'])
            ending = tokenized_ids[-1]
            tokenized_ids = tokenized_ids[:-(len(tokenized_ids) - max_length + 1)] + [ending]
            
        # following the bert convention for calculating segment ids                                                                            
        segment_ids = [0] * (len(question) + 2) + [1] * (len(new_tokens) - 1)
        # mask ids                                                                                            
        mask_ids = [1] * len(tokenized_ids)
        
         # padding
        if len(tokenized_ids) < max_length:
            # Zero-pad up to the sequence length.
            #padding = [tokenizer.pad_token_id] * (max_length - len(tokenized_ids))
            zero_padding = [0] * (max_length - len(tokenized_ids))
            tokenized_ids += zero_padding
            mask_ids += zero_padding
            segment_ids += zero_padding
        assert len(tokenized_ids) == max_length

        if graph_construction == 0 or graph_construction == 1:
            p_doc = nlp(" ".join(v['context']))
            # for graph construction
            p_edges = get_edges(p_doc)

            #p_edges = [(elem[0], elem[1], elem[2], elem[3]) for elem in p_edges]
            q_edges = [(elem[0], elem[1], elem[2]) for elem in q_edges] 

            # bpe와 spacy를 align 하겠다
            _alignment = align_bpe_to_words(tokenizer, new_tokens[1:-1], p_doc)
            # bpe 첫번째만 사용하겠다. 앞의 <s> question </s> </s> length 생각해서.
            bpe_to_node = [align_idx[0] + len(question) + 2 for align_idx in _alignment]
            # question도 align
            _alignment_q = align_bpe_to_words(tokenizer, question, q_doc)
            bpe_to_node = [align_idx[0] + 1 for align_idx in _alignment_q] + bpe_to_node

            pos_labels = [tagger_map[token.tag_] for token in p_doc]
            dep_labels = [depper_map[token.dep_] for token in p_doc]
            parser_tag_map = [p_edge[0] for p_edge in p_edges]

            # add question length for numgnn
            p_edges = [(elem[0]+max_question_length, elem[1]+max_question_length, elem[2]) for elem in p_edges] 

            edges = q_edges + p_edges
            assert len(bpe_to_node) == len(edges)
            
            verb_edges = []
            len_edges = len(edges)
            for edge_idx1 in range(len_edges):
                for edge_idx2 in range(edge_idx1+1, len_edges):
                    if edges[edge_idx1][2] == 1 and edges[edge_idx2][2] == 1:
                        verb_edges.append((edges[edge_idx1][0], edges[edge_idx2][0], 1))  
                        verb_edges.append((edges[edge_idx2][0], edges[edge_idx1][0], 1))            

            edges += verb_edges
        elif graph_construction == 2: # same mention
            p_doc = nlp(" ".join(v['context']))
            # for graph construction
            p_edges = get_edges(p_doc)

            # bpe와 spacy를 align 하겠다
            _alignment = align_bpe_to_words(tokenizer, new_tokens[1:-1], p_doc)
            # bpe 첫번째만 사용하겠다. 앞의 <s> question </s> </s> length 생각해서.
            bpe_to_node = [align_idx[0] + len(question) + 2 for align_idx in _alignment]
            # question도 align
            _alignment_q = align_bpe_to_words(tokenizer, question, q_doc)
            bpe_to_node = [align_idx[0] + 1 for align_idx in _alignment_q] + bpe_to_node

            pos_labels = [tagger_map[token.tag_] for token in p_doc]
            dep_labels = [depper_map[token.dep_] for token in p_doc]
            parser_tag_map = [p_edge[0] for p_edge in p_edges]

            # add question length for numgnn
            p_edges = [(elem[0]+max_question_length, elem[1]+max_question_length, elem[2], elem[3]) for elem in p_edges] 

            edges = q_edges + p_edges
            assert len(bpe_to_node) == len(edges)
            
            # 지금 문제: 다른 sentence에 있는 건 못본다?

            other_edges = []
            for edge_idx1 in range(len(edges)):
                # https://machinelearningknowledge.ai/tutorial-on-spacy-part-of-speech-pos-tagging/                
                for edge_idx2 in range(edge_idx1+1, len(edges)):
                    #if edge_idx1 == edge_idx2: continue                    
                    # connect with sentences
                    if edges[edge_idx1][3].dep_ == 'ROOT' and edges[edge_idx2][3].dep_ == 'ROOT':
                        other_edges.append((edges[edge_idx1][0], edges[edge_idx2][0]))
                        other_edges.append((edges[edge_idx2][0], edges[edge_idx1][0]))                
                    if edges[edge_idx1][3].lemma_ == edges[edge_idx2][3].lemma_ and \
                        edges[edge_idx1][3].pos_ in ["NOUN", "VERB", "PRON", "PROPN", "ADJ", "AUX", "NUM"] :
                        other_edges.append((edges[edge_idx1][0], edges[edge_idx2][0]))
                        other_edges.append((edges[edge_idx2][0], edges[edge_idx1][0]))

            edges = [(edge[0], edge[1]) for edge in edges] + other_edges

        q_event_word_idx, q_tr_word_idx = [], []

        # construct a sample
        # offset: </s> </s> counted in orig_to_tok_map already, so only need to worry about <s>
        if end_to_end:
            # duplicate P + Q for each answer
            labels, offsets = [], []
            events =[]
            for kk, vv in enumerate(v['answers']['labels']):
                labels.append(vv)
                offsets.append(orig_to_tok_map[kk] + len(question) + 1)
            for vv in v['answers']['types']:
                events.append(vv)
            # offsets == (idx of head_masks == 1 )[question len:]
            sample = {'label': labels,
                      'events': events,
                      'offset': offsets,
                      'input_ids': tokenized_ids,
                      'mask_ids': mask_ids,
                      'segment_ids': segment_ids,
                      'question_id': k,
                      'q_head_mask' : bpe_to_node,
                      'question_len' : len(_alignment_q),
                      "edges": edges,
                      "q_event_word_idx" : q_event_word_idx,
                      "q_tr_word_idx" : q_tr_word_idx,
                      "pos_labels": pos_labels,
                      "dep_labels": dep_labels,
                      "parser_tag_map" : parser_tag_map
                      }
            # add these three field for qualitative analysis                                            
            if evaluation:
                sample['passage'] = v['context']
                sample['question'] = v['question']
                sample['question_cluster'] = v['question_cluster']
                sample['cluster_size'] = v['cluster_size']
                sample['answer'] = v['answers']
                sample['individual_answers'] = [a['labels'] for a in v['individual_answers']]
            
            samples.append(sample)
        else:
            # no duplicate P + Q
            labels, offsets = [], []
            events = []
            for vv in v['answers'].values():
                labels.append(vv['label'])
                offsets.append(orig_to_tok_map[vv['idx']] + len(question) + 1)
            for vv in v['answers']['types']:
                events.append(vv)
            sample = {'label': labels,
                      'events': events,
                      'offset': offsets,
                      'input_ids': tokenized_ids,
                      'mask_ids': mask_ids,
                      'segment_ids': segment_ids,
                      'question_id': k,
                      'q_head_mask' : bpe_to_node,
                      'question_len' : len(q_edges),
                      "edges": edges,
                      "q_event_word_idx" : q_event_word_idx,
                      "q_tr_word_idx" : q_tr_word_idx,
                      "pos_labels": pos_labels,
                      "dep_labels": dep_labels,
                      "parser_tag_map" : parser_tag_map
                      }
            
            # add these three field for qualitative analysis         
            if evaluation:
                sample['passage'] = v['context']
                sample['question'] = v['question']
                sample['answer'] = v['answers']
                sample['question_cluster'] = v['question_cluster']
                sample['cluster_size'] = v['cluster_size']
                individual_answers = []
                for vv in v['individual_answers']:
                    individual_answers.append([a['label'] for a in vv.values()])
                sample['individual_answers'] = individual_answers
                
            samples.append(sample)
            
        # check some example data
        if counter < 0:
            print(sample)
        counter += 1
    print("Maximum length after tokenization is: % s" % (max_len_global))

    if not evaluation:
        pickle.dump(samples, open(f"train_{tokenizer.__class__.__name__}_{graph_construction}_{mask_what}_{add_space}_{max_length}_{max_question_length}.pkl", 'wb'))
    return samples

def convert_to_features_deberta_graph2_no_label(data, tokenizer, max_length=150, max_question_length=35, evaluation=False, graph_construction = 0):
    # each sample will have <s> Question </s> </s> Context </s>
    samples = []
    counter = 0
    max_len_global = 0  # to show global max_len without truncating
    
    def get_edges(doc):
        if graph_construction == 0:
            edges = [(token.i, token.head.i, int(token.dep_=="ROOT"), token) for token in doc]
        elif graph_construction == 1:
            edges = [(token.i, token.head.i, int(token.dep_=="ROOT" or token.pos_=="VERB"), token) for token in doc]
        else: 
            edges = [(token.i, token.head.i, int(token.dep_=="ROOT"), token) for token in doc]
        return edges

    for k, v in tqdm(data.items()):
        counter += 1
        
        start_token = ['[CLS]']
        question = tokenizer.tokenize(v['question'])
        q_doc = nlp(v['question'])
        q_edges = get_edges(q_doc)
        
        new_tokens = ["[SEP]"]  # two sent sep symbols according the huggingface documentation
        orig_to_tok_map = []
        for i, token in enumerate(v['context']):
            orig_to_tok_map.append(len(new_tokens))
            # if i == 0 or any(s in token for s in string.punctuation):
            #     temp_tokens = tokenizer.tokenize(token)
            # else:
            #     temp_tokens = tokenizer.tokenize(" " + token)
            temp_tokens = tokenizer.tokenize(token)
            new_tokens.extend(temp_tokens)

        new_tokens.append("[SEP]")
        length = len(new_tokens)
        orig_to_tok_map.append(length)
        assert len(orig_to_tok_map) == len(v['context']) + 1  # account for ending </s>

        tokenized_ids = tokenizer.convert_tokens_to_ids(start_token + question + new_tokens)
        if len(tokenized_ids) > max_len_global:
            max_len_global = len(tokenized_ids)

        if len(tokenized_ids) > max_length:
            ending = tokenized_ids[-1]
            tokenized_ids = tokenized_ids[:-(len(tokenized_ids) - max_length + 1)] + [ending]

        segment_ids = [0] * (len(question) + 2) + [1] * (len(new_tokens) - 1)
        # mask ids
        mask_ids = [1] * len(tokenized_ids)

        # padding
        if len(tokenized_ids) < max_length:
            # Zero-pad up to the sequence length.
            padding = [0] * (max_length - len(tokenized_ids))
            tokenized_ids += padding
            mask_ids += padding
            segment_ids += padding
        assert len(tokenized_ids) == max_length


        if graph_construction == 2: # same mention
            p_doc = nlp(" ".join(v['context']))
            # for graph construction
            p_edges = get_edges(p_doc)

            # bpe와 spacy를 align 하겠다
            _alignment = align_bpe_to_words(tokenizer, new_tokens[1:-1], p_doc)
            # bpe 첫번째만 사용하겠다. 앞의 <s> question </s> </s> length 생각해서.
            bpe_to_node = [align_idx[0] + len(question) + 2 for align_idx in _alignment]
            # question도 align
            _alignment_q = align_bpe_to_words(tokenizer, question, q_doc)
            bpe_to_node = [align_idx[0] + 1 for align_idx in _alignment_q] + bpe_to_node

            pos_labels = [tagger_map[token.tag_] for token in p_doc]
            dep_labels = [depper_map[token.dep_] for token in p_doc]
            parser_tag_map = [p_edge[0] for p_edge in p_edges]

            # add question length for numgnn
            p_edges = [(elem[0]+max_question_length, elem[1]+max_question_length, elem[2], elem[3]) for elem in p_edges] 

            edges = q_edges + p_edges
            assert len(bpe_to_node) == len(edges)
            
            other_edges = []
            for edge_idx1 in range(len(edges)):
                # https://machinelearningknowledge.ai/tutorial-on-spacy-part-of-speech-pos-tagging/                
                for edge_idx2 in range(edge_idx1+1, len(edges)):
                    #if edge_idx1 == edge_idx2: continue                    
                    # connect with sentences
                    if edges[edge_idx1][3].dep_ == 'ROOT' and edges[edge_idx2][3].dep_ == 'ROOT':
                        other_edges.append((edges[edge_idx1][0], edges[edge_idx2][0]))
                        other_edges.append((edges[edge_idx2][0], edges[edge_idx1][0]))                
                    if edges[edge_idx1][3].lemma_ == edges[edge_idx2][3].lemma_ and \
                        edges[edge_idx1][3].pos_ in ["NOUN", "VERB", "PRON", "PROPN", "ADJ", "AUX", "NUM"] :
                        other_edges.append((edges[edge_idx1][0], edges[edge_idx2][0]))
                        other_edges.append((edges[edge_idx2][0], edges[edge_idx1][0]))

            edges = [(edge[0], edge[1]) for edge in edges] + other_edges

        q_event_word_idx, q_tr_word_idx = [], []
        # construct a sample
        # offset: </s> </s> counted in orig_to_tok_map already, so only need to worry about <s>

        # duplicate P + Q for each answer
        offsets = []
        for kk, vv in enumerate(v['context']):
            offsets.append(orig_to_tok_map[kk] + len(question) + 1)
        # if k.startswith("docid_APW20000401.0150_sentid_10"):
        #     print(k)
        sample = {'offset': offsets,
                  'input_ids': tokenized_ids,
                  'mask_ids': mask_ids,
                  'segment_ids': segment_ids,
                  'question_id': k,
                    'q_head_mask' : bpe_to_node,
                    'question_len' : len(_alignment_q),
                    "edges": edges,
                    "q_event_word_idx" : q_event_word_idx,
                    "q_tr_word_idx" : q_tr_word_idx,
                    "pos_labels": pos_labels,
                    "dep_labels": dep_labels,
                    "parser_tag_map" : parser_tag_map
                  }
        # add these three field for qualitative analysis
        if evaluation:
            sample['passage'] = v['context']
            sample['question'] = v['question']
            sample['question_cluster'] = v['question_cluster']
            sample['cluster_size'] = v['cluster_size']
        samples.append(sample)

        # check some example data
        if counter < 0:
            print(sample)
        

    print("Maximum length after tokenization is: % s" % (max_len_global))
    return samples



def align_bpe_to_words(tokenizer, bpe_tokens, other_tokens: List[str]):
    """
    Helper to align GPT-2 BPE to other tokenization formats (e.g., spaCy).
    Args:
        roberta (RobertaHubInterface): RoBERTa instance
        bpe_tokens (torch.LongTensor): GPT-2 BPE tokens of shape `(T_bpe)`
        other_tokens (List[str]): other tokens of shape `(T_words)`
    Returns:
        List[str]: mapping from *other_tokens* to corresponding *bpe_tokens*.
    """
    def clean(text):
        return text.strip()

    # remove whitespaces to simplify alignment
    #bpe_tokens = [roberta.task.source_dictionary.string([x]) for x in bpe_tokens] # convert to string
    bpe_tokens = [
        clean(tokenizer.convert_tokens_to_string(x) if x not in {"<s>", "", "[CLS]"} else x) for x in bpe_tokens
    ]
    other_tokens = [clean(str(o)) for o in other_tokens]

    assert "".join(bpe_tokens) == "".join(other_tokens)

    # create alignment from every word to a list of BPE tokens
    alignment = []
    #bpe_toks = filter(lambda item: item[1] != "", enumerate(bpe_tokens))
    bpe_toks = enumerate(bpe_tokens)
    j, bpe_tok = next(bpe_toks)
    for other_tok in other_tokens:
        bpe_indices = []
        while True:
            if other_tok.startswith(bpe_tok):
                bpe_indices.append(j)
                other_tok = other_tok[len(bpe_tok) :]
                try:
                    j, bpe_tok = next(bpe_toks)
                except StopIteration:
                    j, bpe_tok = None, None
            elif bpe_tok.startswith(other_tok):
                # other_tok spans multiple BPE tokens
                bpe_indices.append(j)
                bpe_tok = bpe_tok[len(other_tok) :]
                other_tok = ""
            else:
                raise Exception('Cannot align "{}" and "{}"'.format(other_tok, bpe_tok))
            if other_tok == "":
                break
        assert len(bpe_indices) > 0
        alignment.append(bpe_indices)
    assert len(alignment) == len(other_tokens)

    return alignment

def align_bert_to_words(tokenizer, bert_tokens, other_tokens: List[str]):
    """
    Helper to align BERT tokens to other tokenization formats (e.g., spaCy).
    Args:
        tokenizer (PreTrainedTokenizer): BERT tokenizer
        bert_tokens (List[str]): BERT tokens of shape `(T_bert)`
        other_tokens (List[str]): other tokens of shape `(T_words)`
    Returns:
        List[List[int]]: mapping from every word in *other_tokens* to a list of corresponding BERT subword indices.
    """
    def clean(text):
        return text.strip()

    # remove whitespaces to simplify alignment
    bert_tokens = [clean(str(t)).replace("##", "").lower() for t in bert_tokens]
    other_tokens = [clean(str(o)).lower() for o in other_tokens]

    assert "".join(bert_tokens) == "".join(other_tokens)

    # create alignment from every word to a list of BPE tokens
    alignment = []
    bert_toks = enumerate(bert_tokens)
    j, bert_tok = next(bert_toks)
    for other_tok in other_tokens:
        bert_indices = []
        while True:
            if other_tok.startswith(bert_tok):
                bert_indices.append(j)
                other_tok = other_tok[len(bert_tok) :]
                try:
                    j, bert_tok = next(bert_toks)
                except StopIteration:
                    j, bert_tok = None, None
            elif bert_tok.startswith(other_tok):
                # other_tok spans multiple BPE tokens
                bert_indices.append(j)
                bert_tok = bert_tok[len(other_tok) :]
                other_tok = ""
            else:
                raise Exception('Cannot align "{}" and "{}"'.format(other_tok, bert_tok))
            if other_tok == "":
                break
        assert len(bert_indices) > 0
        alignment.append(bert_indices)
    assert len(alignment) == len(other_tokens)

    return alignment
