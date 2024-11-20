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
import string
from glob import glob
import random
import os

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
nlp = spacy.load("en_core_web_sm")
tagger_map = {elem : i for i, elem in enumerate(nlp.get_pipe('tagger').labels)}
depper_map = {elem : i for i, elem in enumerate(nlp.get_pipe('parser').labels)}

def load_data(data_dir, split, suffix, train_ratio=1):
    filename = "%s%s%s" % (data_dir, split, suffix)
    print("==========load data from %s ===========" % filename)
    with open(filename, "r") as read_file:
        return_file = json.load(read_file)
        #TODO: erase this when debug done!
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


def convert_to_features_roberta_no_label(data, tokenizer, max_length=150, evaluation=False):
    # each sample will have <s> Question </s> </s> Context </s>
    samples = []
    counter = 0
    max_len_global = 0  # to show global max_len without truncating

    for k, v in data.items():
        start_token = ['<s>']
        question = tokenizer.tokenize(v['question'])

        new_tokens = ["</s>", "</s>"]  # two sent sep symbols according the huggingface documentation
        orig_to_tok_map = []
        for i, token in enumerate(v['context']):
            orig_to_tok_map.append(len(new_tokens))
            temp_tokens = tokenizer.tokenize(token)
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

        # construct a sample
        # offset: </s> </s> counted in orig_to_tok_map already, so only need to worry about <s>

        # duplicate P + Q for each answer
        offsets = []
        for kk, vv in enumerate(v['context']):
            offsets.append(orig_to_tok_map[kk] + len(question) + 1)

        sample = {'offset': offsets,
                  'input_ids': tokenized_ids,
                  'mask_ids': mask_ids,
                  'segment_ids': segment_ids,
                  'question_id': k}
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



def convert_to_features_no_label(data, tokenizer, max_length=150, evaluation=False):
    # each sample will have <s> Question </s> </s> Context </s>
    samples = []
    counter = 0
    max_len_global = 0  # to show global max_len without truncating

    for k, v in data.items():
        start_token = ['[CLS]']
        question = tokenizer.tokenize(v['question'])

        new_tokens = ["[SEP]"]  # two sent sep symbols according the huggingface documentation
        orig_to_tok_map = []
        for i, token in enumerate(v['context']):
            orig_to_tok_map.append(len(new_tokens))
            temp_tokens = tokenizer.tokenize(token)
            new_tokens.extend(temp_tokens)

        new_tokens.append("[SEP]")
        length = len(new_tokens)
        orig_to_tok_map.append(length)
        assert len(orig_to_tok_map) == len(v['context']) + 1  # account for ending </s>

        segment_ids = [0] * (len(question) + 2) + [1] * (len(new_tokens) - 1)
        # mask ids
        mask_ids = [1] * len(segment_ids)

        tokenized_ids = tokenizer.convert_tokens_to_ids(start_token + question + new_tokens)
        if len(tokenized_ids) > max_len_global:
            max_len_global = len(tokenized_ids)

        if len(tokenized_ids) > max_length:
            ending = tokenized_ids[-1]
            tokenized_ids = tokenized_ids[:-(len(tokenized_ids) - max_length + 1)] + [ending]

        # padding
        if len(tokenized_ids) < max_length:
            # Zero-pad up to the sequence length.
            padding = [0] * (max_length - len(tokenized_ids))
            tokenized_ids += padding
            mask_ids += padding
            segment_ids += padding
        assert len(tokenized_ids) == max_length

        # construct a sample
        # offset: </s> </s> counted in orig_to_tok_map already, so only need to worry about <s>

        # duplicate P + Q for each answer
        offsets = []
        for kk, vv in enumerate(v['context']):
            offsets.append(orig_to_tok_map[kk] + len(question) + 1)

        sample = {'offset': offsets,
                  'input_ids': tokenized_ids,
                  'mask_ids': mask_ids,
                  'segment_ids': segment_ids,
                  'question_id': k}
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


def mask_what(question): # what 만 가려보자 안복잡하게..
    # for start_string in ["What events", "What event", "What", "what events", "what event", "what"]:
    #     if question.startswith(start_string):
    #         question = question.replace(start_string, "<mask>", 1)
    #         return question
    # #  "When the judge's ruling comes, what will have already happened?",
    # for mid_string in ["what events", "what event", "what"]:
    #     if "what" in question:
    #             question = question.replace("what", "<mask>", 1)
    if question[0].lower() == "what":
            question[0] = "<mask>"
            return question, 0
    #  ex) "When the judge's ruling comes, what will have already happened?",
    elif "Ġwhat" in question:
            what_index = question.index("Ġwhat")
            question[what_index] = "<mask>"
            return question, what_index
    # ex) "How did the men get to victory during the event?"
    else: 
        return question, -1
    


def convert_to_features_roberta(data, tokenizer, max_length=150, evaluation=False,
                                instance=True, end_to_end=False, label_is_event=False, mask_question=0):

    # if not evaluation and os.path.exists(f"train_roberta.pkl"):
    #     print("load data from pickle. If you revised data reader, delete this script")
    #     samples =pickle.load(open(f"train_roberta.pkl", 'rb'))
    #     return samples

    # each sample will have <s> Question </s> </s> Context </s>
    samples = []
    counter = 0
    max_len_global = 0 # to show global max_len without truncating 


    for k, v in tqdm(data.items()):
        segment_ids = []
        start_token = ['<s>']
        question = tokenizer.tokenize(v['question'])
        # if mask_question:
        #     question, mask_index = mask_what(question)
        question_mask = [0] + [1] * len(question)

        new_tokens = ["</s>", "</s>"] # two sent sep symbols according the huggingface documentation
        orig_to_tok_map = []
        for i, token in enumerate(v['context']):
            orig_to_tok_map.append(len(new_tokens))
            temp_tokens = tokenizer.tokenize(token)
            new_tokens.extend(temp_tokens)
            
        new_tokens.append("</s>")
        length = len(new_tokens)
        orig_to_tok_map.append(length)
        assert len(orig_to_tok_map) == len(v['context']) + 1 # account for ending </s>

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
            padding = [0] * (max_length - len(tokenized_ids))
            tokenized_ids += padding
            mask_ids += padding
            segment_ids += padding
            question_mask += [0] * (max_length - len(question_mask))
        assert len(tokenized_ids) == max_length

        # construct a sample
        # offset: </s> </s> counted in orig_to_tok_map already, so only need to worry about <s>
        if end_to_end:
            # duplicate P + Q for each answer
            labels, offsets = [], []
            events = []
            if label_is_event:
                for kk, vv in enumerate(v['answers']['types']):
                    labels.append(vv)
                    offsets.append(orig_to_tok_map[kk] + len(question) + 1)
            else:
                for kk, vv in enumerate(v['answers']['labels']):
                    labels.append(vv)
                    offsets.append(orig_to_tok_map[kk] + len(question) + 1)
            # these exist in train, dev set!
            for vv in v['answers']['types']:
                events.append(vv)
            sample = {'label': labels,
                      'events': events,
                      'offset': offsets,
                      'input_ids': tokenized_ids,
                      'mask_ids': mask_ids,
                      'segment_ids': segment_ids,
                      'question_id': k,
                      'question_mask' : question_mask}
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
            events=[]
            if label_is_event:
                for vv in v['answers'].values():
                    labels.append(vv['types'])
                    offsets.append(orig_to_tok_map[vv['idx']] + len(question) + 1)
            else:
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
                      'question_mask' : question_mask}
            
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

    # if not evaluation:
    #     pickle.dump(samples, open(f"train_roberta.pkl", 'wb'))

    return samples

def convert_to_features_roberta_event(data, tokenizer, max_length=150, evaluation=False,
                                instance=True, end_to_end=False, label_is_event=False):

    # each sample will have <s> Question </s> </s> Context </s>
    samples = []
    counter = 0
    max_len_global = 0 # to show global max_len without truncating 
    
    id_list = list()

    for k, v in tqdm(data.items()):
        k = "_".join(k.split("_")[:-1])
        if k in id_list:
            continue
        else:
            id_list.append(k)

        segment_ids = []
        start_token = ['<s>']
        new_tokens = [] # two sent sep symbols according the huggingface documentation
        orig_to_tok_map = []
        for i, token in enumerate(v['context']):
            orig_to_tok_map.append(len(new_tokens))
            temp_tokens = tokenizer.tokenize(token)
            new_tokens.extend(temp_tokens)
            
        new_tokens.append("</s>")
        length = len(new_tokens)
        orig_to_tok_map.append(length)
        assert len(orig_to_tok_map) == len(v['context']) + 1 # account for ending </s>

        tokenized_ids = tokenizer.convert_tokens_to_ids(start_token + new_tokens)
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

        p_doc = nlp(" ".join(v['context']))
        pos_labels = [tagger_map[token.tag_] for token in p_doc]
        dep_labels = [depper_map[token.dep_] for token in p_doc]

        # bpe와 spacy를 align 하겠다
        _alignment = align_bpe_to_words(tokenizer, new_tokens[:-1], p_doc)
        # bpe 첫번째만 사용하겠다. 앞의 <s> length 생각해서.
        bpe_to_node = [align_idx[0] + 1 for align_idx in _alignment]
        # construct a sample
        # offset: </s> </s> counted in orig_to_tok_map already, so only need to worry about <s>
        if end_to_end:
            # duplicate P + Q for each answer
            labels, offsets = [], []
            events = []
            if label_is_event:
                for kk, vv in enumerate(v['answers']['types']):
                    labels.append(vv)
                    offsets.append(orig_to_tok_map[kk])
            # these exist in train, dev set!
            for vv in v['answers']['types']:
                events.append(vv)
            sample = {'label': labels,
                      'events': events,
                      'offset': offsets,
                      'input_ids': tokenized_ids,
                      'mask_ids': mask_ids,
                      'segment_ids': segment_ids,
                      'question_id': k,
                      'pos_labels': pos_labels,
                      'dep_labels': dep_labels,
                      'bpe_to_node': bpe_to_node}
            # add these three field for qualitative analysis                                            
            samples.append(sample)
        else:
            # no duplicate P + Q
            labels, offsets = [], []
            events=[]
            if label_is_event:
                for vv in v['answers'].values():
                    labels.append(vv['types'])
                    offsets.append(orig_to_tok_map[vv['idx']])
            for vv in v['answers']['types']:
                events.append(vv)
            sample = {'label': labels,
                      'events': events,
                      'offset': offsets,
                      'input_ids': tokenized_ids,
                      'mask_ids': mask_ids,
                      'segment_ids': segment_ids,
                      'question_id': k,
                      'pos_labels': pos_labels,
                      'dep_labels': dep_labels,
                      'bpe_to_node': bpe_to_node}
                
            samples.append(sample)
            
        # check some example data
        if counter < 0:
            print(sample)
        counter += 1

    print("Maximum length after tokenization is: % s" % (max_len_global))

    return samples


def convert_to_features_bart(data, tokenizer, max_length=150, evaluation=False,
                                instance=True, end_to_end=False, label_is_event=False):
    
    # each sample will have <s> Question </s> </s> Context </s>
    samples = []
    counter = 0
    max_len_global = 0 # to show global max_len without truncating 


    for k, v in tqdm(data.items()):
        segment_ids = []
        start_token = ['<s>']
        question = tokenizer.tokenize(v['question'])
        question_mask = [0] + [1] * len(question)

        new_tokens = ["</s>", "</s>"] # two sent sep symbols according the huggingface documentation
        orig_to_tok_map = []
        for i, token in enumerate(v['context']):
            orig_to_tok_map.append(len(new_tokens))
            temp_tokens = tokenizer.tokenize(token)
            new_tokens.extend(temp_tokens)
            
        new_tokens.append("</s>")
        length = len(new_tokens)
        orig_to_tok_map.append(length)
        assert len(orig_to_tok_map) == len(v['context']) + 1 # account for ending </s>

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
            padding = [0] * (max_length - len(tokenized_ids))
            tokenized_ids += [tokenizer.pad_token_id] * len(padding)
            mask_ids += padding
            segment_ids += padding
            question_mask += [0] * (max_length - len(question_mask))
        assert len(tokenized_ids) == max_length

        # construct a sample
        # offset: </s> </s> counted in orig_to_tok_map already, so only need to worry about <s>
        if end_to_end:
            # duplicate P + Q for each answer
            labels, offsets = [], []
            events = []
            if label_is_event:
                for kk, vv in enumerate(v['answers']['types']):
                    labels.append(vv)
                    offsets.append(orig_to_tok_map[kk] + len(question) + 1)
            else:
                for kk, vv in enumerate(v['answers']['labels']):
                    labels.append(vv)
                    offsets.append(orig_to_tok_map[kk] + len(question) + 1)
            # these exist in train, dev set!
            for vv in v['answers']['types']:
                events.append(vv)
            sample = {'label': labels,
                      'events': events,
                      'offset': offsets,
                      'input_ids': tokenized_ids,
                      'mask_ids': mask_ids,
                      'segment_ids': segment_ids,
                      'question_id': k,
                      'question_mask' : question_mask}
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
            events=[]
            if label_is_event:
                for vv in v['answers'].values():
                    labels.append(vv['types'])
                    offsets.append(orig_to_tok_map[vv['idx']] + len(question) + 1)
            else:
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
                      'question_mask' : question_mask}
            
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

    return samples


def convert_to_features_roberta_qg(data, tokenizer, max_length=150, evaluation=False,
                                instance=True, end_to_end=False, semi_struct_files = None):

    # if not evaluation and os.path.exists(f"train_roberta_qg.pkl"):
    #     print("load data from pickle. If you revised data reader, delete this script")
    #     samples =pickle.load(open(f"train_roberta_qg.pkl", 'rb'))
    #     return samples

    semi_struct = json.load(open(semi_struct_files[0]))
    semi_struct = dict(**semi_struct, **json.load(open(semi_struct_files[1])))       

    def _get_semi_struct(k):
        structure = []
        try:
            qa_pairs = semi_struct["_".join(k.split("_")[:-1])]
        except KeyError:
            print("key error: ", "_".join(k.split("_")[:-1]))
            return []
        for aq in qa_pairs:
            aq = aq[0]
            structure += tokenizer.tokenize(" ".join((aq[1], aq[0])))
            structure += ["</s>"]
        return structure

    # each sample will have <s> Question </s> </s> Context </s>
    samples = []
    counter = -3
    max_len_global = 0 # to show global max_len without truncating 


    for k, v in tqdm(data.items()):
        segment_ids = []
        start_token = ['<s>']
        question = tokenizer.tokenize(v['question'])
        question_mask = [0] + [1] * len(question)

        new_tokens = ["</s>", "</s>"] # two sent sep symbols according the huggingface documentation
        orig_to_tok_map = []
        for i, token in enumerate(v['context']):
            orig_to_tok_map.append(len(new_tokens))
            temp_tokens = tokenizer.tokenize(token)
            new_tokens.extend(temp_tokens)
            
        new_tokens.append("</s>")
        length = len(new_tokens)
        orig_to_tok_map.append(length)
        assert len(orig_to_tok_map) == len(v['context']) + 1 # account for ending </s>

        semi_struct_tokens = _get_semi_struct(k)

        tokenized_ids = tokenizer.convert_tokens_to_ids(start_token + question + new_tokens + semi_struct_tokens)
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
            padding = [0] * (max_length - len(tokenized_ids))
            tokenized_ids += padding
            mask_ids += padding
            segment_ids += padding
        question_mask += [0] * (max_length - len(question_mask))
        assert len(tokenized_ids) == max_length

        # construct a sample
        # offset: </s> </s> counted in orig_to_tok_map already, so only need to worry about <s>
        if end_to_end:
            # duplicate P + Q for each answer
            labels, offsets = [], []
            events = []
            for kk, vv in enumerate(v['answers']['labels']):
                labels.append(vv)
                offsets.append(orig_to_tok_map[kk] + len(question) + 1)
            # these exist in train, dev set!
            for vv in v['answers']['types']:
                events.append(vv)
            sample = {'label': labels,
                      'events': events,
                      'offset': offsets,
                      'input_ids': tokenized_ids,
                      'mask_ids': mask_ids,
                      'segment_ids': segment_ids,
                      'question_id': k,
                      'question_mask' : question_mask}
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
            events=[]
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
                      'question_mask' : question_mask}
            
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
            print(semi_struct_tokens)
            print(sample)
        counter += 1

    print("Maximum length after tokenization is: % s" % (max_len_global))

    # if not evaluation:
    #     pickle.dump(samples, open(f"train_roberta_qg.pkl", 'wb'))

    return samples


def convert_to_features_roberta_graph(data, tokenizer, max_length=150, max_question_length = 0, evaluation=False,
                                instance=True, end_to_end=False, is_qdgat=0, suffix="", graph_construction = 0, event_chain_files=None):
    # if not evaluation and os.path.exists(f"train_graph_{is_qdgat}_{suffix}.pkl"):
    #     print("load data from pickle. If you revised data reader, delete this script")
    #     samples =pickle.load(open(f"train_graph_{is_qdgat}_{suffix}.pkl", 'rb'))
    #     return samples
    # each sample will have <s> Question </s> </s> Context </s>
    samples = []
    counter = 0
    max_len_global = 0 # to show global max_len without truncating 

    def get_edges(doc, tokens=None, is_passage = False):
        """get edges -> convert spacy edge index to original(nltk) index"""
        # ('before', 'mark', 'SCONJ') # t.text, t.dep_, t.pos_
        #edges = [(token.i, token.head.i, int(token.pos_ == "VERB" or token.dep_=="ROOT")) for token in doc]
        if graph_construction == 0:
            edges = [(token.i, token.head.i, int(token.dep_=="ROOT")) for token in doc]
        elif graph_construction == 1:
            edges = [(token.i, token.head.i, int(token.dep_=="ROOT" or token.pos_=="VERB")) for token in doc]
        if is_passage:
            # sanitize
            if len(tokens) != len(doc): # mismatch between tokenizer result
                extra_offsets = []
                i=0
                j=0
                while i < len(doc):
                    if tokens[j] == doc[i].text:
                        extra_offsets.append(extra_offsets[-1]) if len(extra_offsets) > 0 else extra_offsets.append(0)
                        i+=1
                        j+=1
                    else:
                        extra_offsets.append(extra_offsets[-1]) if len(extra_offsets) > 0 else extra_offsets.append(0)
                        temp = 1
                        temp_tokens = doc[i].text + doc[i+temp].text
                        while tokens[j] != temp_tokens:
                            temp+=1
                            temp_tokens+=doc[i+temp].text
                            extra_offsets.append(extra_offsets[-1]-1)
                        #end
                        i+=temp+1
                        extra_offsets.append(extra_offsets[-1]-1)
                        j+=1
                assert len(extra_offsets) == len(doc)
                edges = [(elem[0]+extra_offsets[elem[0]], elem[1]+extra_offsets[elem[1]], elem[2]) for elem in edges]
            assert max([elem[0] for elem in edges]) == len(tokens)-1, print(edges, tokens, [tok.text for tok in doc])
            return edges                
        else: # question
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
        segment_ids = []
        start_token = ['<s>']
        head_mask = [0]
        question = tokenizer.tokenize(v['question'])
        q_doc = nlp(v['question'])
        q_edges = get_edges(q_doc)
        alignment = align_bpe_to_words(tokenizer, question, q_doc)
        # get head mask # first token of word
        _head_idx = set([elem[0] for elem in alignment])
        head_mask += [1 if i in _head_idx else 0 for i, _ in enumerate (question)]
        # align edges
        _extra_offsets = [0]
        for i in range(1, len(alignment)):
            if alignment[i][0] == alignment[i-1][0]:
                _extra_offsets.append(_extra_offsets[-1]-1)
            else: 
                _extra_offsets.append(_extra_offsets[-1])
        # if spacy token is shorter or more
        #if 1 in _extra_offsets or -1 in _extra_offsets: # "isn", "'t" #
        #    print()
        q_edges = [(elem[0]+_extra_offsets[elem[0]], elem[1]+_extra_offsets[elem[1]], elem[2]) for elem in q_edges]
        # get question len
        question_len = sum(head_mask)
        assert max([elem[0] for elem in q_edges])+1 == sum(head_mask)        
        
        new_tokens = ["</s>", "</s>"] # two sent sep symbols according the huggingface documentation
        orig_to_tok_map = []
        head_mask += [0, 0]
        for i, token in enumerate(v['context']):
            orig_to_tok_map.append(len(new_tokens))
            temp_tokens = tokenizer.tokenize(token)
            head_mask += [1]
            head_mask += [0] * (len(temp_tokens)-1)
            new_tokens.extend(temp_tokens)
            
        new_tokens.append("</s>")
        length = len(new_tokens)
        orig_to_tok_map.append(length)
        assert len(orig_to_tok_map) == len(v['context']) + 1 # account for ending </s>

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
            p_edges = get_edges(p_doc, v["context"], is_passage=True)

            pos_labels = [tagger_map[token.tag_] for token in p_doc]
            dep_labels = [depper_map[token.dep_] for token in p_doc]
            parser_tag_map = [p_edge[0] for p_edge in p_edges]

            # add question length for numgnn
            if not is_qdgat:
                p_edges = [(elem[0]+max_question_length, elem[1]+max_question_length, elem[2]) for elem in p_edges] 

            if is_qdgat:
                edges = p_edges
            else: # gcn
                # join q & p edges + offset -> unified graph
                edges = q_edges + p_edges
            
            verb_edges = []
            len_edges = len(edges)
            for edge_idx1 in range(len_edges):
                for edge_idx2 in range(edge_idx1+1, len_edges):
                    if edges[edge_idx1][-1] == 1 and edges[edge_idx2][-1] == 1:
                        verb_edges.append((edges[edge_idx1][0], edges[edge_idx2][0], 1))  
                        verb_edges.append((edges[edge_idx2][0], edges[edge_idx1][0], 1))            

            edges += verb_edges

        elif graph_construction == 3: # syntactic q + semantic p

            pass

        # pad
        if len(head_mask) < max_length:
            head_mask += [0] * (max_length - len(head_mask))

        q_event_word_idx, q_tr_word_idx = q_tr_event_rulebase(q_doc, p_doc)         
        q_event_word_idx= [idx + _extra_offsets[idx] for idx in q_event_word_idx]
        q_tr_word_idx= [idx + _extra_offsets[idx] for idx in q_tr_word_idx]
        
        # temporal relation mask
        q_event_word_idx += [-1] * (max_question_length - len(q_event_word_idx))
        # event mask
        q_tr_word_idx += [-1] * (max_question_length - len(q_tr_word_idx))
        
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
                      'q_head_mask' : head_mask,
                      'question_len' : question_len,
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
                      'q_head_mask' : head_mask,
                      'question_len' : question_len,
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

    # if not evaluation:
    #     pickle.dump(samples, open(f"train_graph_{is_qdgat}_{suffix}.pkl", 'wb'))

    return samples

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

def flatten_answers_no_label(answers):
    offsets = [a for ans in answers for a in ans]
    lengths = [len(ans) for ans in answers]

    assert len(offsets) == sum(lengths)
    return offsets, lengths


def convert_to_features(data, tokenizer, max_length=150, evaluation=False,
                        instance=True, end_to_end=False):
    # each sample will have [CLS] + Question + [SEP] + Context                                                                                 
    samples = []
    counter = 0
    max_len_global = 0 # to show global max_len without truncating                                                                             
    for k, v in data.items():
        segment_ids = []
        start_token = ['[CLS]']
        question = tokenizer.tokenize(v['question'])

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
                      'question_id': k}
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
                      'question_id': k}

            # add these three field for qualitative analysis            
            if evaluation:
                sample['passage'] = v['context']
                sample['question'] = v['question']
                sample['answer'] = v['answers']
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
        counter += 1

    print("Maximum length after tokenization is: % s" % (max_len_global))
    return samples


def convert_to_features_roberta_qmask(data, tokenizer, max_length=150, evaluation=False,
                                instance=True, end_to_end=False, label_is_event=False):

    # if not evaluation and os.path.exists(f"train_roberta.pkl"):
    #     print("load data from pickle. If you revised data reader, delete this script")
    #     samples =pickle.load(open(f"train_roberta.pkl", 'rb'))
    #     return samples

    # each sample will have <s> Question </s> </s> Context </s>
    samples = []
    counter = 0
    max_len_global = 0 # to show global max_len without truncating 


    for k, v in tqdm(data.items()):

        # tmp_k = "_".join(k.split("_")[1:-3])
        # if tmp_k in timemls:
        #     print('skip ',k)
        #     continue

        segment_ids = []
        start_token = ['<s>']
        question = tokenizer.tokenize(v['question'])
        question_mask = [0] + [1] * len(question)

        new_tokens = ["</s>", "</s>"] # two sent sep symbols according the huggingface documentation
        orig_to_tok_map = []
        for i, token in enumerate(v['context']):
            orig_to_tok_map.append(len(new_tokens))
            temp_tokens = tokenizer.tokenize(token)
            new_tokens.extend(temp_tokens)
            
        new_tokens.append("</s>")
        length = len(new_tokens)
        orig_to_tok_map.append(length)
        assert len(orig_to_tok_map) == len(v['context']) + 1 # account for ending </s>

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
            padding = [0] * (max_length - len(tokenized_ids))
            tokenized_ids += padding
            mask_ids += padding
            segment_ids += padding
            question_mask += [0] * (max_length - len(question_mask))
        assert len(tokenized_ids) == max_length

        # construct a sample
        # offset: </s> </s> counted in orig_to_tok_map already, so only need to worry about <s>
        if end_to_end:
            # duplicate P + Q for each answer
            labels, offsets = [], []
            events = []
            if label_is_event:
                for kk, vv in enumerate(v['answers']['types']):
                    labels.append(vv)
                    offsets.append(orig_to_tok_map[kk] + len(question) + 1)
            else:
                for kk, vv in enumerate(v['answers']['labels']):
                    labels.append(vv)
                    offsets.append(orig_to_tok_map[kk] + len(question) + 1)
            # these exist in train, dev set!
            for vv in v['answers']['types']:
                events.append(vv)
            sample = {'label': labels,
                      'events': events,
                      'offset': offsets,
                      'input_ids': tokenized_ids,
                      'mask_ids': mask_ids,
                      'segment_ids': segment_ids,
                      'question_id': k,
                      'question_mask' : question_mask}
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
            events=[]
            if label_is_event:
                for vv in v['answers'].values():
                    labels.append(vv['types'])
                    offsets.append(orig_to_tok_map[vv['idx']] + len(question) + 1)
            else:
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
                      'question_mask' : question_mask}
            
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

    # if not evaluation:
    #     pickle.dump(samples, open(f"train_roberta.pkl", 'wb'))

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
        clean(tokenizer.convert_tokens_to_string(x) if x not in {"<s>", ""} else x) for x in bpe_tokens
    ]
    other_tokens = [clean(str(o)) for o in other_tokens]

    assert "".join(bpe_tokens) == "".join(other_tokens), print(bpe_tokens, other_tokens)

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


TRS = set(["before", "befure", "after", "while", "future", "past", "during", "when", "prior", "now",  
        "following", "since", "until", "the same time", "result of", "end of", "once", "the time of", "between",
        " lead to ", " led to ", " if ", " as "]) 
        #"in the future as a result of" "before after"  #TODO: how to catch this one? 
TR_VERB_NOUN = {"have", "happen", "what", "start", "begin", "finish", "event", "end", "lead", "future", "past", "result", "time", "while"}
INTERROGATIVES = {"what", "how", "why", "when", "who"}
def q_tr_event_rulebase(question, passage):
    """_summary_

    Args:
        data (dict): json.load(open(processed_data_file))
    """
    to_reverse=False
    not_to_reverse = False
    # first, get prefix of question
    min_idx = 10000
    tr_found = ""
    found=False
    for tr in TRS: 
        idx = question.text.find(tr)
        if idx != -1:
            if min_idx > idx:
                # min = result of, idx = in the future
                if min_idx < idx + len(tr) + 5 : # do nothing
                    continue
                else:
                    min_idx = idx
                    tr_found = tr
                    found=True
            # min = in the future, idx = result of
            elif min_idx + len(tr_found) + 5 > idx :  # TODO: "What will happen in the future as a result of number 16 Moore Street gaining significant national importance?"
                min_idx = idx
                tr_found = tr
                found=True
    if found:
        tok_i = -1
        for tok in question:
            if tok.idx < min_idx + len(tr_found) :
                tok_i = tok.i
            else:
                break
        q_tr1 = question[:tok_i+1]
        q_event = question[tok_i+1 : -1] # remove PUNC
        # if the question is reversed:  ex. "The group fired two mortar shells after what happened to the group's commander",
        for interrogative in INTERROGATIVES:
            if interrogative in q_event.text.lower():
                to_reverse=True
                break
        for interrogative in INTERROGATIVES:
            if interrogative in q_tr1.text.lower():
                not_to_reverse=True
                break
        if to_reverse and not not_to_reverse:
            q_tr1 = question[tok_i:-1]
            q_event = question[:tok_i]
        q_tr = list(question[:tok_i+1])
        q_event_word_idx = list(range(tok_i+1, len(question)-1))
    else:
        q_tr = list(question[:-1]) # delete "?"
        q_event_word_idx = list()
    # second, delete substring which is in context
    c_words = set([tok.lemma_.lower() for tok in passage])
    q_event_is = []
    for tok in q_tr:
        if tok.lemma_.lower() in c_words and \
            tok.pos_ in {"NOUN", "VERB", "PROPN", "PRON", "ADJ", "NUM"} and \
            tok.lemma_ not in TR_VERB_NOUN:
            q_event_is.append(tok.i)
    if len(q_event_is) > 0:
        q_event_word_idx.extend(list(range(min(q_event_is), max(q_event_is)+1)))
    q_tr_word_idx = [tok.i for tok in q_tr if (tok.i not in q_event_word_idx or tok.lemma_.lower() != "what")]
    
    return q_event_word_idx, q_tr_word_idx