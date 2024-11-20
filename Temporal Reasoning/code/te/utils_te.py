from typing import Iterator, List, Mapping, Union, Optional, Set
from collections import defaultdict, Counter, OrderedDict
from datetime import datetime
import json
import pickle
import numpy as np
import logging
import string
import spacy
from tqdm import tqdm
from utils import align_bpe_to_words
import torch
import random
from torch.utils.data import Dataset
import math
import os
import re

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
nlp = spacy.load("en_core_web_lg")

class Document:
    def __init__(self, data, dataname, ignore_nonetype=False, istrain=True):
        self.id = data["fid"]
        self.text = data["text"]
        self.events = data["events"] 
        self.event_span_map = {e["eiid"]: e["offset"] for e in self.events if e.get("eiid", None) is not None}
        self.event_sentid_map = {e["eiid"]: e["sent_id"] for e in self.events if e.get("eiid", None) is not None}
        self.relations = data["relations"]
        self.dataname = dataname.lower()

        self.pair_labels = []
        self.pair_ids = []
        self.pair_spans = []
        self.pair_texts = []
        self.pair_sents = []

        self.get_labels(ignore_nonetype, istrain)
    
    # def sort_events(self):
    #     self.events = sorted(self.events, key=lambda x: (x["sent_id"], x["offset"][0]))
    #     self.sorted_event_spans = [(event["sent_id"], event["offset"]) for event in self.events]

    def get_labels(self, ignore_nonetype, istrain):
        rel2id = matres_label_map
        inverse_rel = matres_inverse_rel
        pair2rel = {}
        for rel in self.relations:
            for pair in self.relations[rel]:
                pair2rel[tuple(pair)] = rel2id[rel]
            
        for pair, label in pair2rel.items():
            if pair[0] not in self.event_span_map or pair[1] not in self.event_span_map:
                continue
            self.pair_ids.append(pair)
            self.pair_labels.append(label)
            pair_sents = " ".join(self.text[self.event_sentid_map[pair[0]]:self.event_sentid_map[pair[1]]+1])
            self.pair_sents.append(pair_sents)
            offset_right = len(" ".join(self.text[self.event_sentid_map[pair[0]]:self.event_sentid_map[pair[1]]]))
            spans = (self.event_span_map[pair[0]], (self.event_span_map[pair[1]]))
            # map character to words(separated by space)
            spans = ((spans[0][0], spans[0][1]), (spans[1][0] + offset_right, spans[1][1] + offset_right))
            spans = (self.char_span_to_word_span(self.pair_sents[-1], spans[0]), 
                     self.char_span_to_word_span(self.pair_sents[-1], spans[1]))
            self.pair_spans.append(spans)
            self.pair_texts.append((self.text[self.event_sentid_map[pair[0]]][self.event_span_map[pair[0]][0]: self.event_span_map[pair[0]][1]], 
                                    self.text[self.event_sentid_map[pair[1]]][self.event_span_map[pair[1]][0]: self.event_span_map[pair[1]][1]]))
        
        # assert same length
        assert len(self.pair_ids) == len(self.pair_labels)
        assert len(self.pair_ids) == len(self.pair_spans)
        assert len(self.pair_ids) == len(self.pair_texts)
        assert len(self.pair_ids) == len(self.pair_sents)

    def char_span_to_word_span(self, text, char_span):
        # Split the text into words
        words = text.split()
        # Initialize variables
        start_char, end_char = char_span
        # remove spaces
        while text[start_char] == " ":
            start_char += 1
        r = re.compile(r'\S+|\b\w+\b')
        words = {i :(m.start(0), m.group(0)) for i, m in enumerate(r.finditer(text))}
        assert len(words) == len(text.split())
        for i in range (len(words)):
            if words[i][0] == start_char:
                start_word = i
                end_word = i + 1

        # assert words[start_word][0] == start_char
        return start_word, end_word


def select_field(data, field):
    # collect a list of field in data                                                                  
    # fields: 'label', 'offset', 'input_ids, 'mask_ids', 'segment_ids', 'question_id'                            
    return [ex[field] for ex in data]

def select_field_te(features, field):
    return [
        [
            choice[field]
            for choice in feature.choices_features
        ]
        for feature in features
    ]

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


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


def cal_f1(pred_labels, true_labels, label_map, log=False):
    def safe_division(numr, denr, on_err=0.0):
        return on_err if denr == 0.0 else numr / denr

    assert len(pred_labels) == len(true_labels)

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
    f1_score = safe_division(2.0 * precision * recall, precision + recall)
    if log:
        logger.info("Correct: %d\tTrue: %d\tPred: %d" % (n_correct, n_true, n_pred))
        logger.info("Overall Precision: %.4f\tRecall: %.4f\tF1: %.4f" % (precision, recall, f1_score))
    return f1_score


tbd_label_map = OrderedDict([('VAGUE', 'VAGUE'),
                             ('BEFORE', 'BEFORE'),
                             ('AFTER', 'AFTER'),
                             ('SIMULTANEOUS', 'SIMULTANEOUS'),
                             ('INCLUDES', 'INCLUDES'),
                             ('IS_INCLUDED', 'IS_INCLUDED'),
                             ])

matres_label_map = OrderedDict([('VAGUE', 'VAGUE'),
                                ('BEFORE', 'BEFORE'),
                                ('AFTER', 'AFTER'),
                                ('EQUAL', 'EQUAL')
                                ])

red_label_map = OrderedDict([
    #('TERMINATES', 'TERMINATES'),
    ('SIMULTANEOUS', 'OVERLAP'),
    #('REINITIATES',  'REINITIATES'),
    ('OVERLAP/PRECONDITION', 'OVERLAP'),
    ('OVERLAP/CAUSES', 'OVERLAP'),
    ('OVERLAP', 'OVERLAP'),
    #('INITIATES', 'INITIATES'),
    ('ENDS-ON', 'BEFORE'),
    #('CONTINUES', 'CONTINUES'),
    ('CONTAINS-SUBEVENT', 'INCLUDES'),
    ('CONTAINS', 'INCLUDES'),
    #('IS_INCLUDED', 'IS_INCLUDED'),
    ('BEGINS-ON', 'OVERLAP'),
    ('BEFORE/PRECONDITION', 'BEFORE'),
    ('BEFORE/CAUSES', 'BEFORE'),
    ('BEFORE', 'BEFORE'),
    #('AFTER', 'AFTER')
])

matres_inverse_rel = {
        "BEFORE": "AFTER",
        "AFTER": "BEFORE"
    }

class Event():
    def __init__(self, id, type, text, tense, polarity, span):
        self.id = id
        self.type = type
        self.text = text
        self.tense = tense
        self.polarity = polarity
        self.span = span

class TEFeatures(object):
    def __init__(self,
                 example_id,
                 length,
                 doc_id,
                 features,
                 label,
                 temp_group_ind,
                 edge_info,
                 token_idxs,
                 p_text, # to distinguish pairs
                 ):
        self.example_id = example_id
        self.length = length
        self.doc_id = doc_id
        self.choices_features = [
            {
                'input_ids': features[0], # 
                'input_mask': features[1],  #
                'segment_ids': features[2], # 
                'left_id': features[3],
                'right_id': features[4],
                'lidx_s': features[5], # 
                'lidx_e': features[6],
                'ridx_s': features[7], # 
                'ridx_e': features[8], 
                'pred_ind': features[9],
                'sample_counter': features[10] #
            }
        ]
        self.label = label # 
        self.temp_group_ind = temp_group_ind
        self.edge_info = edge_info
        self.token_idxs = token_idxs
        self.p_text = p_text

temp_groups = [['before', 'previous to', 'prior to', 'preceding', 'followed', 'until'],
               ['after', 'following', 'since', 'soon after', 'once', 'now that'],
               ['during', 'while', 'when', 'the same time', 'at the time', 'meanwhile'],
               ['earlier', 'previously', 'formerly', 'in the past', 'yesterday', 'last time'],
               ['consequently', 'subsequently', 'in turn', 'henceforth', 'later', 'then'],
               ['initially', 'originally', 'at the beginning', 'to begin', 'starting with', 'to start with'],
               ['finally', 'in the end', 'at last', 'lastly']]

temp_groups_names = ['before', 'after', 'during', 'past', 'future', 'begin', 'end']

def get_edges(doc):
    edges = [(token.i, token.head.i, int(token.dep_=="ROOT"), token) for token in doc]
    return edges

def convert_examples_to_features_te(data_dir, data_type, split, tokenizer, max_seq_length,
                                            is_training, includeIds=None, analyze=False, max_question_length = 3):
    if split=="train":
        if data_type == "matres":
            save_path_name = f"{tokenizer.__class__.__name__}_{max_seq_length}_{max_question_length}_{split}_{data_type}_{includeIds[0]}.pkl"
        elif data_type == "tbd":
            save_path_name = f"{tokenizer.__class__.__name__}_{max_seq_length}_{max_question_length}_{split}_{data_type}.pkl"
        print(save_path_name)
    if split=="train" and os.path.exists(save_path_name):
        print("load data from pickle. If you revised data reader, delete this script")
        samples =pickle.load(open(save_path_name, 'rb'))
        return samples

    """Loads a data file into a list of InputBatch"""
    if data_type == "matres":
        label_map = matres_label_map
    elif data_type == "tbd":
        label_map = tbd_label_map
    elif data_type == "red":
        label_map = red_label_map
    if 'roberta' in tokenizer.__class__.__name__.lower():
        add_space = True
    else:
        add_space = False

    all_labels = list(OrderedDict.fromkeys(label_map.keys()))
    label_to_id = OrderedDict([(all_labels[l], l) for l in range(len(all_labels))])
    id_to_label = OrderedDict([(l, all_labels[l]) for l in range(len(all_labels))])

    print(label_to_id)
    print(id_to_label)
    examples = pickle.load(open("%s/%s/%s.pickle" % (data_dir, data_type, split), "rb"))
    count, counter, global_max = 0, 0, 0
    features, lengths = [], []
    label_counter = Counter()
    category_counter = Counter()
    for ex_id, ex in tqdm(examples.items()):
        # bidirectional loss + transformer seems to tackle the reverse pair issue
        if '_rev' in ex['rel_type']:
            label = ex['rel_type'][:-4]
        else:
            label = ex['rel_type']
        if label not in label_map:
            # print(label)
            continue

        label_counter[label] += 1

        label_id = label_to_id[label]
        doc_id = ex['doc_id']

        # handle train / dev for matres
        if includeIds and doc_id not in includeIds:
            continue

        left_id = ex['left_event'].id
        right_id = ex['right_event'].id

        pos_dict = ex['doc_dictionary']

        group_ind = [0]*8
        if analyze:
            n_match = 0
            for v in pos_dict.values():
                for i, group in enumerate(temp_groups):
                    if v[0].lower() in group:
                        ti = i
                        n_match += 1
                        group_ind[ti] = 1
                        category_counter[temp_groups_names[ti]] += 1

            if n_match == 0:
                group_ind[-1] = 1
                category_counter['None'] += 1

        all_keys, lidx_start, lidx_end, ridx_start, ridx_end = token_idx(ex['left_event'].span,
                                                                         ex['right_event'].span,
                                                                         pos_dict)
        left_seq = [pos_dict[x][0] for x in all_keys[:lidx_start]]
        right_seq = [pos_dict[x][0] for x in all_keys[ridx_end + 1:]]
        in_seq = [pos_dict[x][0] for x in all_keys[lidx_start:ridx_end + 1]]

        try:
            sent_start = max(loc for loc, val in enumerate(left_seq) if val == '.') + 1
        except:
            sent_start = 0

        try:
            sent_end = ridx_end + 1 + min(loc for loc, val in enumerate(right_seq) if val == '.')
        except:
            sent_end = len(pos_dict)

        assert sent_start < sent_end
        assert sent_start <= lidx_start
        assert ridx_end <= sent_end

        # if > 2 sentences, not predicting
        pred_ind = True

        sent_key = all_keys[sent_start:sent_end]
        orig_sent = [pos_dict[x][0].lower() for x in sent_key]

        lidx_start_s = lidx_start - sent_start
        lidx_end_s = lidx_end - sent_start
        ridx_start_s = ridx_start - sent_start
        ridx_end_s = ridx_end - sent_start

        # event_tokens

        orig_to_tok_map = []

        # question is two events.
        new_tokens = [tokenizer.cls_token]
        question = []
        question.extend(tokenizer.tokenize(" "+ex['left_event'].text.lower()))
        question.extend(tokenizer.tokenize(" "+ex['right_event'].text.lower()))
        new_tokens.extend(question)
        new_tokens.extend([tokenizer.sep_token, tokenizer.sep_token])
        q_doc = nlp(" " + ex['left_event'].text.lower() + " " + ex['right_event'].text.lower())
        q_edges = [(token.i, token.i, 1, token) for token in q_doc]

        for i, token in enumerate(orig_sent):
            orig_to_tok_map.append(len(new_tokens))
            if add_space:
                if i == 0 or any(s in token for s in string.punctuation):
                    temp_tokens = tokenizer.tokenize(token)
                else:
                    temp_tokens = tokenizer.tokenize(" " + token)
            else:
                temp_tokens = tokenizer.tokenize(token)
            #temp_tokens = tokenizer.tokenize(token)
            new_tokens.extend(temp_tokens)
            # for t in temp_tokens:
            #     mask_ids.append(1)

        new_tokens_before_pad = new_tokens[3+len(question):]

        length = len(new_tokens)
        orig_to_tok_map.append(length)
        new_tokens.append(tokenizer.sep_token)
        assert len(orig_to_tok_map) == len(orig_sent) + 1
        
        

        # append ending
        mask_ids = [1] * len(new_tokens)
        assert len(mask_ids) == len(new_tokens)
        # FIXME: if not roberta, it needs token type ids

        length += 1
        if length > global_max:
            global_max = length
        # truncate if token lenght exceed max and set pred_ind = False
        if len(new_tokens) + 1 > max_seq_length:
            new_tokens = new_tokens[:max_seq_length - 1]
            mask_ids = mask_ids[:max_seq_length - 1]
            pred_ind = False
            length = max_seq_length
        lengths.append(length)

        # padding
        new_tokens += [tokenizer.pad_token] * (max_seq_length - len(new_tokens))
        mask_ids += [0] * (max_seq_length - len(mask_ids))

        # map original token index into bert (word_piece) index
        lidx_start_ss = orig_to_tok_map[lidx_start_s]
        lidx_end_ss = orig_to_tok_map[lidx_end_s + 1] - 1

        ridx_start_ss = orig_to_tok_map[ridx_start_s]
        ridx_end_ss = orig_to_tok_map[ridx_end_s + 1] - 1

        ## a quick trick to tackle long sents: use last token
        if pred_ind == False:
            orig_to_tok_map_reverse = {v: k for k, v in enumerate(orig_to_tok_map)}
            if ridx_end_ss >= max_seq_length:
                ridx_start_ss = max_seq_length - 1
                ridx_end_ss = max_seq_length - 1
                ridx_start_ss_tmp = ridx_start_ss
                while ridx_start_ss_tmp > 0 and orig_to_tok_map_reverse.get(ridx_start_ss_tmp, 0) == 0:
                    ridx_start_ss_tmp -= 1
                ridx_start_s = orig_to_tok_map_reverse[ridx_start_ss_tmp]
                count += 1
            if lidx_end_ss >= max_seq_length:
                lidx_start_ss = max_seq_length - 1
                lidx_end_ss = max_seq_length - 1
                lidx_start_ss_tmp = lidx_start_ss
                while lidx_start_ss_tmp > 0 and orig_to_tok_map_reverse.get(lidx_start_ss_tmp, 0) == 0:
                    lidx_start_ss_tmp -= 1
                lidx_start_s = orig_to_tok_map_reverse[lidx_start_ss_tmp]


        input_ids = tokenizer.convert_tokens_to_ids(new_tokens)
        # only one segment for Roberta
        segments_ids = [0] * len(input_ids)

        assert len(input_ids) == len(segments_ids)
        assert len(input_ids) == len(mask_ids)

        """make gcn features before padding"""
        p_doc = nlp(" ".join(orig_sent))
        # for graph construction
        p_edges = get_edges(p_doc)
        # bpe와 spacy를 align 하겠다
        _alignment = align_bpe_to_words(tokenizer, new_tokens_before_pad, p_doc)
        # bpe 첫번째만 사용하겠다. 앞의 <s> question </s> </s> length 생각해서.
        bpe_to_node = [align_idx[0] + len(question) + 3 for align_idx in _alignment]
        # question도 align
        _alignment_q = align_bpe_to_words(tokenizer, question, q_doc)
        bpe_to_node = [align_idx[0] + 1 for align_idx in _alignment_q] + bpe_to_node

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
                if edges[edge_idx1][2] and edges[edge_idx2][2] == 'ROOT':
                    other_edges.append((edges[edge_idx1][0], edges[edge_idx2][0]))
                    other_edges.append((edges[edge_idx2][0], edges[edge_idx1][0]))                
                if edges[edge_idx1][3].lemma_ == edges[edge_idx2][3].lemma_ and \
                    edges[edge_idx1][3].pos_ in ["NOUN", "VERB", "PRON", "PROPN", "ADJ", "AUX", "NUM"] :
                    other_edges.append((edges[edge_idx1][0], edges[edge_idx2][0]))
                    other_edges.append((edges[edge_idx2][0], edges[edge_idx1][0]))

        edges = [(edge[0], edge[1]) for edge in edges] + other_edges

        edge_info = dict(edges = edges,
                         bpe_to_node = bpe_to_node,
                         question_len = len(_alignment_q)
                         )
        token_idxs = dict(offsets = orig_to_tok_map, lidx_s = lidx_start_s, ridx_s = ridx_start_s)

        features.append(TEFeatures(ex_id, length, doc_id,
                                   (input_ids, mask_ids, segments_ids, left_id, right_id,
                                    lidx_start_ss, lidx_end_ss, ridx_start_ss, ridx_end_ss, pred_ind, counter),
                                    label_id, group_ind, edge_info, token_idxs, " ".join(orig_sent)),
                        )
        counter += 1
    print(label_counter)
    print(category_counter)

    print("%d sentences have more than %d tokens, %d" % (
        sum([v for k, v in Counter(lengths).items() if k >= max_seq_length]), max_seq_length, count))
    print("TE global max: %s" % global_max)

    if split=="train":
        pickle.dump(features, open(save_path_name, 'wb'))

    return features


def convert_examples_to_features_te_matres(data_dir, data_type, split, tokenizer, max_seq_length,
                                            is_training, includeIds=None, analyze=False, max_question_length = 3):
    add_space = True
    if split=="train":
        if data_type == "matres":
            save_path_name = f"{tokenizer.__class__.__name__}_{max_seq_length}_{max_question_length}_{split}_{data_type}_{includeIds[0]}.pkl"
        elif data_type == "tbd":
            save_path_name = f"{tokenizer.__class__.__name__}_{max_seq_length}_{max_question_length}_{split}_{data_type}.pkl"
        print(save_path_name)
    if split=="train" and os.path.exists(save_path_name):
        print("load data from pickle. If you revised data reader, delete this script")
        samples =pickle.load(open(save_path_name, 'rb'))
        return samples
    """Loads a data file into a list of InputBatch"""
    label_map = matres_label_map

    all_labels = list(OrderedDict.fromkeys(label_map.keys()))
    label_to_id = OrderedDict([(all_labels[l], l) for l in range(len(all_labels))])
    id_to_label = OrderedDict([(l, all_labels[l]) for l in range(len(all_labels))])

    print(label_to_id)
    print(id_to_label)

    data_dir = "/root/temporal_reasoning/MAVEN-ERE/data/processed" 
    examples = [json.loads(line) for line in open("%s/%s/%s.json" % (data_dir, data_type.upper(), split), "r").readlines()]
    # ['fid', 'title', 'text', 'events', 'timexes', 'relations']
    count, counter, global_max = 0, 0, 0
    features, lengths = [], []
    label_counter = Counter()
    category_counter = Counter()
    group_ind = [0] * 8
    label_map = matres_label_map
    all_labels = list(OrderedDict.fromkeys(label_map.keys()))
    label_to_id = OrderedDict([(all_labels[l], l) for l in range(len(all_labels))])
    id_to_label = OrderedDict([(l, all_labels[l]) for l in range(len(all_labels))])

    for ex_id, ex in tqdm(enumerate(examples)):
        ex = Document(ex, data_type)
        doc_id = ex.id
        for i, label_id in enumerate(ex.pair_labels):
            label_id = label_to_id[label_id]
            orig_sent = ex.pair_sents[i].split()
            left_event = Event(ex.pair_ids[i][0], None, 
                               ex.pair_texts[i][0], None, None, 
                               ex.pair_spans[i][0])
            right_event = Event(ex.pair_ids[i][1], None,
                                ex.pair_texts[i][1], None, None,
                                ex.pair_spans[i][1])
            left_id = left_event.id
            right_id = right_event.id
            # event_tokens
            orig_to_tok_map = []

            # question is two events.
            new_tokens = [tokenizer.cls_token]
            question = []
            question.extend(tokenizer.tokenize(" "+left_event.text.lower()))
            question.extend(tokenizer.tokenize(" "+right_event.text.lower()))
            new_tokens.extend(question)
            new_tokens.extend([tokenizer.sep_token, tokenizer.sep_token])
            q_doc = nlp(" " + left_event.text.lower() + " " + right_event.text.lower())
            q_edges = [(token.i, token.i, 1, token) for token in q_doc]
            
            for i, token in enumerate(orig_sent):
                orig_to_tok_map.append(len(new_tokens))
                if add_space:
                    if i == 0:
                        temp_tokens = tokenizer.tokenize(token)
                    else:
                        temp_tokens = tokenizer.tokenize(" " + token)
                else:
                    temp_tokens = tokenizer.tokenize(token)
                #temp_tokens = tokenizer.tokenize(token)
                new_tokens.extend(temp_tokens)
                # for t in temp_tokens:

            new_tokens_before_pad = new_tokens[3+len(question):]

            length = len(new_tokens)
            orig_to_tok_map.append(length)
            new_tokens.append(tokenizer.sep_token)
            assert len(orig_to_tok_map) == len(orig_sent) + 1
            
            # append ending
            mask_ids = [1] * len(new_tokens)
            assert len(mask_ids) == len(new_tokens)
            # FIXME: if not roberta, it needs token type ids

            length += 1
            if length > global_max:
                global_max = length
            pred_ind = True
            # truncate if token lenght exceed max and set pred_ind = False
            if len(new_tokens) + 1 > max_seq_length:
                new_tokens = new_tokens[:max_seq_length - 1]
                mask_ids = mask_ids[:max_seq_length - 1]
                pred_ind = False
                length = max_seq_length
            lengths.append(length)

            # padding
            new_tokens += [tokenizer.pad_token] * (max_seq_length - len(new_tokens))
            mask_ids += [0] * (max_seq_length - len(mask_ids))

            # map original token index into bert (word_piece) index
            lidx_start_s = left_event.span[0]
            lidx_end_s = left_event.span[1]
            lidx_start_ss = orig_to_tok_map[lidx_start_s]
            lidx_end_ss = orig_to_tok_map[lidx_end_s]

            ridx_start_s = right_event.span[0]
            ridx_end_s = right_event.span[1]
            ridx_start_ss = orig_to_tok_map[ridx_start_s]
            ridx_end_ss = orig_to_tok_map[ridx_end_s]

            ## a quick trick to tackle long sents: use last token
            if pred_ind == False:
                orig_to_tok_map_reverse = {v: k for k, v in enumerate(orig_to_tok_map)}
                if ridx_end_ss >= max_seq_length:
                    ridx_start_ss = max_seq_length - 1
                    ridx_end_ss = max_seq_length - 1
                    ridx_start_ss_tmp = ridx_start_ss
                    while ridx_start_ss_tmp > 0 and orig_to_tok_map_reverse.get(ridx_start_ss_tmp, 0) == 0:
                        ridx_start_ss_tmp -= 1
                    ridx_start_s = orig_to_tok_map_reverse[ridx_start_ss_tmp]
                    count += 1
                if lidx_end_ss >= max_seq_length:
                    lidx_start_ss = max_seq_length - 1
                    lidx_end_ss = max_seq_length - 1
                    lidx_start_ss_tmp = lidx_start_ss
                    while lidx_start_ss_tmp > 0 and orig_to_tok_map_reverse.get(lidx_start_ss_tmp, 0) == 0:
                        lidx_start_ss_tmp -= 1
                    lidx_start_s = orig_to_tok_map_reverse[lidx_start_ss_tmp]


            input_ids = tokenizer.convert_tokens_to_ids(new_tokens)
            # only one segment for Roberta
            segments_ids = [0] * len(input_ids)

            assert len(input_ids) == len(segments_ids)
            assert len(input_ids) == len(mask_ids)

            """make gcn features before padding"""
            p_doc = nlp(" ".join(orig_sent))
            # for graph construction
            p_edges = get_edges(p_doc)
            # bpe와 spacy를 align 하겠다
            _alignment = align_bpe_to_words(tokenizer, new_tokens_before_pad, p_doc)
            # bpe 첫번째만 사용하겠다. 앞의 <s> question </s> </s> length 생각해서.
            bpe_to_node = [align_idx[0] + len(question) + 3 for align_idx in _alignment]
            # question도 align
            _alignment_q = align_bpe_to_words(tokenizer, question, q_doc)
            bpe_to_node = [align_idx[0] + 1 for align_idx in _alignment_q] + bpe_to_node

            assert all([bpe < len(new_tokens_before_pad) + 3 + len(question) for bpe in bpe_to_node])

            # add question length for numgnn
            p_edges = [(elem[0]+max_question_length, elem[1]+max_question_length, elem[2], elem[3]) for elem in p_edges] 

            edges = q_edges + p_edges
            assert len(bpe_to_node) == len(edges)

            if pred_ind == False:
                truncated_edges = []
                truncated_bpe_to_node = []
                for edge_idx, edge in enumerate(edges):
                    if edge[0] < max_seq_length and edge[1] < max_seq_length:
                        truncated_edges.append(edge)
                        truncated_bpe_to_node.append(bpe_to_node[edge_idx])
                edges = truncated_edges
                bpe_to_node = truncated_bpe_to_node

            # 지금 문제: 다른 sentence에 있는 건 못본다?

            other_edges = []
            for edge_idx1 in range(len(edges)):
                # https://machinelearningknowledge.ai/tutorial-on-spacy-part-of-speech-pos-tagging/                
                for edge_idx2 in range(edge_idx1+1, len(edges)):
                    #if edge_idx1 == edge_idx2: continue                    
                    # connect with sentences
                    if edges[edge_idx1][2] and edges[edge_idx2][2] == 'ROOT':
                        other_edges.append((edges[edge_idx1][0], edges[edge_idx2][0]))
                        other_edges.append((edges[edge_idx2][0], edges[edge_idx1][0]))                
                    if edges[edge_idx1][3].lemma_ == edges[edge_idx2][3].lemma_ and \
                        edges[edge_idx1][3].pos_ in ["NOUN", "VERB", "PRON", "PROPN", "ADJ", "AUX", "NUM"] :
                        other_edges.append((edges[edge_idx1][0], edges[edge_idx2][0]))
                        other_edges.append((edges[edge_idx2][0], edges[edge_idx1][0]))

            edges = [(edge[0], edge[1]) for edge in edges] + other_edges


            edge_info = dict(edges = edges,
                            bpe_to_node = bpe_to_node,
                            question_len = len(_alignment_q)
                            )
            token_idxs = dict(offsets = orig_to_tok_map, lidx_s = lidx_start_s, ridx_s = ridx_start_s)

            features.append(TEFeatures(ex_id, length, doc_id,
                                    (input_ids, mask_ids, segments_ids, left_id, right_id,
                                        lidx_start_ss, lidx_end_ss, ridx_start_ss, ridx_end_ss, pred_ind, counter),
                                        label_id, group_ind, edge_info, token_idxs, " ".join(orig_sent)),
                            )
            counter += 1
    print(label_counter)
    print(category_counter)

    print("%d sentences have more than %d tokens, %d" % (
        sum([v for k, v in Counter(lengths).items() if k >= max_seq_length]), max_seq_length, count))
    print("TE global max: %s" % global_max)

    if split=="train":
        pickle.dump(features, open(save_path_name, 'wb'))

    return features

def token_idx(left, right, pos_dict):
    all_keys = list(pos_dict.keys())

    ### to handle case with multiple tokens
    lkey_start = str(left[0])
    lkey_end = str(left[1])

    ### to handle start is not an exact match -- "tomtake", which should be "to take"
    lidx_start = 0
    while int(all_keys[lidx_start].split(':')[1][:-1]) <= left[0]:
        lidx_start += 1

    ### to handle case such as "ACCOUNCED--" or multiple token ends with not match
    lidx_end = lidx_start
    try:
        while left[1] > int(all_keys[lidx_end].split(':')[1][:-1]):
            lidx_end += 1
    except:
        lidx_end -= 1

    ridx_start = 0
    while int(all_keys[ridx_start].split(':')[1][:-1]) <= right[0]:
        ridx_start += 1

    ridx_end = ridx_start
    try:
        while right[1] > int(all_keys[ridx_end].split(':')[1][:-1]):
            ridx_end += 1
    except:
        ridx_end -= 1
    return all_keys, lidx_start, lidx_end, ridx_start, ridx_end

class ClassificationReport:
    def __init__(self, name, true_labels: List[Union[int, str]],
                 pred_labels: List[Union[int, str]]):

        assert len(true_labels) == len(pred_labels)
        self.num_tests = len(true_labels)
        self.total_truths = Counter(true_labels)
        self.total_predictions = Counter(pred_labels)
        self.name = name
        self.labels = sorted(set(true_labels) | set(pred_labels))
        self.confusion_mat = self.confusion_matrix(true_labels, pred_labels)
        self.accuracy = sum(y == y_ for y, y_ in zip(true_labels, pred_labels)) / len(true_labels)
        self.trim_label_width = 15
        self.rel_f1 = 0.0
        self.res_dict = {}

    @staticmethod
    def confusion_matrix(true_labels: List[str], predicted_labels: List[str]) \
            -> Mapping[str, Mapping[str, int]]:
        mat = defaultdict(lambda: defaultdict(int))
        for truth, prediction in zip(true_labels, predicted_labels):
            mat[truth][prediction] += 1
        return mat

    def __repr__(self):
        res = f'Name: {self.name}\t Created: {datetime.now().isoformat()}\t'
        res += f'Total Labels: {len(self.labels)} \t Total Tests: {self.num_tests}\n'
        display_labels = [label[:self.trim_label_width] for label in self.labels]
        label_widths = [len(l) + 1 for l in display_labels]
        max_label_width = max(label_widths)
        header = [l.ljust(w) for w, l in zip(label_widths, display_labels)]
        header.insert(0, ''.ljust(max_label_width))
        res += ''.join(header) + '\n'
        for true_label, true_disp_label in zip(self.labels, display_labels):
            predictions = self.confusion_mat[true_label]
            row = [true_disp_label.ljust(max_label_width)]
            for pred_label, width in zip(self.labels, label_widths):
                row.append(str(predictions[pred_label]).ljust(width))
            res += ''.join(row) + '\n'
        res += '\n'

        def safe_division(numr, denr, on_err=0.0):
            return on_err if denr == 0.0 else numr / denr

        def num_to_str(num):
            return '0' if num == 0 else str(num) if type(num) is int else f'{num:.4f}'

        n_correct = 0
        n_true = 0
        n_pred = 0

        all_scores = []
        header = ['Total  ', 'Predictions', 'Correct', 'Precision', 'Recall  ', 'F1-Measure']
        res += ''.ljust(max_label_width + 2) + '  '.join(header) + '\n'
        head_width = [len(h) for h in header]

        exclude_list = ['None']
        if "matres" in self.name: exclude_list.append('VAGUE')

        for label, width, display_label in zip(self.labels, label_widths, display_labels):
            if label not in exclude_list:
                total_count = self.total_truths.get(label, 0)
                pred_count = self.total_predictions.get(label, 0)

                n_true += total_count
                n_pred += pred_count

                correct_count = self.confusion_mat[label][label]
                n_correct += correct_count

                precision = safe_division(correct_count, pred_count)
                recall = safe_division(correct_count, total_count)
                f1_score = safe_division(2 * precision * recall, precision + recall)
                all_scores.append((precision, recall, f1_score))
                self.res_dict[label] = (f1_score, total_count)
                row = [total_count, pred_count, correct_count, precision, recall, f1_score]
                row = [num_to_str(cell).ljust(w) for cell, w in zip(row, head_width)]
                row.insert(0, display_label.rjust(max_label_width))
                res += '  '.join(row) + '\n'

        # weighing by the truth label's frequency
        label_weights = [safe_division(self.total_truths.get(label, 0), self.num_tests)
                         for label in self.labels if label not in exclude_list]
        weighted_scores = [(w * p, w * r, w * f) for w, (p, r, f) in zip(label_weights, all_scores)]

        assert len(label_weights) == len(weighted_scores)

        res += '\n'
        res += '  '.join(['Weighted Avg'.rjust(max_label_width),
                          ''.ljust(head_width[0]),
                          ''.ljust(head_width[1]),
                          ''.ljust(head_width[2]),
                          num_to_str(sum(p for p, _, _ in weighted_scores)).ljust(head_width[3]),
                          num_to_str(sum(r for _, r, _ in weighted_scores)).ljust(head_width[4]),
                          num_to_str(sum(f for _, _, f in weighted_scores)).ljust(head_width[5])])

        print(n_correct, n_pred, n_true)

        precision = safe_division(n_correct, n_pred)
        recall = safe_division(n_correct, n_true)
        f1_score = safe_division(2.0 * precision * recall, precision + recall)

        res += f'\n Total Examples: {self.num_tests}'
        res += f'\n Overall Precision: {num_to_str(precision)}'
        res += f'\n Overall Recall: {num_to_str(recall)}'
        res += f'\n Overall F1: {num_to_str(f1_score)} '
        self.rel_f1 = f1_score
        return res
    
def flatten_answers(answers):    
    # flatten answers and use batch length to map back to the original input   
    offsets = [a for ans in answers for a in ans[1]]
    labels = [a for ans in answers for a in ans[0]]
    #events = [a for ans in answers for a in ans[2]]
    lengths = [len(ans[0]) for ans in answers]

    assert len(offsets)  == sum(lengths)
    assert len(labels) == sum(lengths)
    #assert len(events) == sum(lengths)
    
    return offsets, labels, lengths


class TeContDataset(Dataset):
    def __init__(self, features, batch_size, evaluation=False, full_passage=False, train_ratio=1):
        feature_sametext_group = defaultdict(list)
        for feature in features:
            # feature_sametext_group[feature.p_text].append(feature)
            feature_sametext_group[feature.p_text + feature.choices_features[0]['left_id']].append(feature)
        self.feature_list = []
        if evaluation:
            self.feature_list = list(feature_sametext_group.values())
        else:                
            for p_text, feature_group in feature_sametext_group.items():
                for _ in range(math.ceil(len(feature_group) / batch_size)):  # ex. 27 instances in one cluster -> ceil(27/6) = 5 same instances
                    self.feature_list.append(feature_group)
    def __len__(self):
        return len(self.feature_list)
    def __getitem__(self, index) :
        return self.feature_list[index]


class GraphDataCollatorTE:
    def __init__(self, device, evaluation=False,
     contrastive_loss=0, n_negs=0, batch_size=1):
        self.is_eval = evaluation
        self.device = device
        self.n_negs = n_negs # if negative sampling
        self.contrastive_loss = bool(contrastive_loss)
        self.batch_size = batch_size

    def __call__(self, batch):
        features = batch[0]
        # if evaluation: use all sentences
        if not self.is_eval:
            random.shuffle(features)
            # cons: cannot see some instance? -> it will be fine if #epochs is large enough 
            features = features[:self.batch_size]

        input_ids = [f.choices_features[0]['input_ids'] for f in features]
        input_masks = [f.choices_features[0]['input_mask'] for f in features] 
        segment_ids = [f.choices_features[0]['segment_ids'] for f in features]
        bpe_to_node = [f.edge_info['bpe_to_node'] for f in features]
        question_len = [f.edge_info['question_len'] for f in features]
        edges = [f.edge_info['edges'] for f in features]
        lidx_s = [f.token_idxs['lidx_s'] for f in features]
        ridx_s = [f.token_idxs['ridx_s'] for f in features]
        offsets = [f.token_idxs['offsets'] for f in features]
        labels = [f.label for f in features]
        counters = [f.choices_features[0]['sample_counter'] for f in features]

        # padding
        pad_len_edges = max([len(elem) for elem in edges])

        # to add parser label info
        #parser_labels = None
        lidx_s_tok = [f.choices_features[0]['lidx_s'] for f in features]
        ridx_s_tok = [f.choices_features[0]['ridx_s'] for f in features]
        for i in range(len(features)):
            assert offsets[i][lidx_s[i]] == lidx_s_tok[i]
            assert offsets[i][ridx_s[i]] == ridx_s_tok[i]

        for i in range(len(features)):
            # pad
            edges[i] += [edges[i][0]] * (pad_len_edges - len(edges[i])) # padding this way doesn't create extra edges

        edges = torch.tensor(edges, dtype=torch.long).to(self.device)
        input_ids = torch.tensor(input_ids, dtype=torch.long).to(self.device)
        input_masks = torch.tensor(input_masks, dtype=torch.long).to(self.device)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long).to(self.device)
        labels = torch.tensor(labels, dtype=torch.long).to(self.device)

        samples = input_ids, input_masks, segment_ids, \
            bpe_to_node, question_len, edges, lidx_s, ridx_s, offsets, \
            labels, counters
        return samples