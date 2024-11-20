from __future__ import absolute_import, division, print_function, unicode_literals

import json
import logging
import math
import os
from pydoc import doc
import sys
from io import open
from itertools import chain
import torch
from torch import nn
import torch.nn.functional as F
from utils import replace_masked_values, depper_map, tagger_map
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, BCELoss, NLLLoss
from transformers import BertPreTrainedModel, BertModel, RobertaModel, BartConfig, BartModel, BartPretrainedModel, RobertaConfig, \
    DebertaConfig, DebertaModel, DebertaPreTrainedModel, ElectraConfig, ElectraPreTrainedModel, ElectraModel, \
    DebertaV2Model, DebertaV2PreTrainedModel
from transformers.models.roberta.modeling_roberta import RobertaLMHead

from datetime import datetime

logger = logging.getLogger(__name__)

# BERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
#     'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin",
#     'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-pytorch_model.bin",
#     'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-pytorch_model.bin",
#     'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-pytorch_model.bin",
#     'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-pytorch_model.bin",
#     'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-pytorch_model.bin",
#     'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-pytorch_model.bin",
#     'bert-base-german-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-cased-pytorch_model.bin",
#     'bert-large-uncased-whole-word-masking': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-pytorch_model.bin",
#     'bert-large-cased-whole-word-masking': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-pytorch_model.bin",
#     'bert-large-uncased-whole-word-masking-finetuned-squad': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-pytorch_model.bin",
#     'bert-large-cased-whole-word-masking-finetuned-squad': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-finetuned-squad-pytorch_model.bin",
#     'bert-base-cased-finetuned-mrpc': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-finetuned-mrpc-pytorch_model.bin",
# }

# ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP = {
#     'roberta-base': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-pytorch_model.bin",
#     'roberta-large': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-pytorch_model.bin",
#     'roberta-large-mnli': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-pytorch_m\
# odel.bin",
# }
        
class MultitaskClassifier(BertPreTrainedModel):
    def __init__(self, config, mlp_hid=16, use_contrastive_loss=0):
        super(MultitaskClassifier, self).__init__(config)
        self.bert = BertModel(config)
        self.num_labels = 2
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear1 = nn.Linear(config.hidden_size, mlp_hid)
        self.linear2 = nn.Linear(mlp_hid, self.num_labels)
        self.act = nn.Tanh()
        self.init_weights()
        self.contrastive_loss = use_contrastive_loss

    def forward(self, input_ids, offsets, lengths, attention_mask=None,
                token_type_ids=None, position_ids=None, head_mask=None, labels=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        outputs = self.dropout(outputs[0])
                                                           
        idx = 0
        vectors = []
        for b, l in enumerate(lengths):
            for i in range(l):
                vectors.append(outputs[b, offsets[idx], :].unsqueeze(0))
                idx += 1
        assert idx == sum(lengths)

        vectors = torch.cat(vectors, dim=0)
        outputs = self.act(self.linear1(vectors))
        logits = self.linear2(outputs)
        
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return logits, loss

        return logits
    

class MultitaskClassifierDeberta(DebertaPreTrainedModel):
    def __init__(self, config, mlp_hid=16, use_contrastive_loss=0):
        super(MultitaskClassifierDeberta, self).__init__(config)
        self.deberta = DebertaModel(config)
        self.num_labels = 2
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear1 = nn.Linear(config.hidden_size, mlp_hid)
        self.linear2 = nn.Linear(mlp_hid, self.num_labels)
        self.act = nn.Tanh()
        self.init_weights()
        self.contrastive_loss = use_contrastive_loss

    def forward(self, input_ids, offsets, lengths, attention_mask=None,
                token_type_ids=None, position_ids=None, head_mask=None, labels=None):
        outputs = self.deberta(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids)

        outputs = self.dropout(outputs[0])
                                                           
        idx = 0
        vectors = []
        for b, l in enumerate(lengths):
            for i in range(l):
                vectors.append(outputs[b, offsets[idx], :].unsqueeze(0))
                idx += 1
        assert idx == sum(lengths)

        vectors = torch.cat(vectors, dim=0)
        outputs = self.act(self.linear1(vectors))
        logits = self.linear2(outputs)
        
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return logits, loss

        return logits

class MultitaskClassifierDebertaV2(DebertaV2PreTrainedModel):
    def __init__(self, config, mlp_hid=16, use_contrastive_loss=0):
        super(MultitaskClassifierDebertaV2, self).__init__(config)
        self.deberta = DebertaV2Model(config)
        self.num_labels = 2
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear1 = nn.Linear(config.hidden_size, mlp_hid)
        self.linear2 = nn.Linear(mlp_hid, self.num_labels)
        self.act = nn.Tanh()
        self.init_weights()
        self.contrastive_loss = use_contrastive_loss

    def forward(self, input_ids, offsets, lengths, attention_mask=None,
                token_type_ids=None, position_ids=None, head_mask=None, labels=None):
        outputs = self.deberta(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids)

        outputs = self.dropout(outputs[0])
                                                           
        idx = 0
        vectors = []
        for b, l in enumerate(lengths):
            for i in range(l):
                vectors.append(outputs[b, offsets[idx], :].unsqueeze(0))
                idx += 1
        assert idx == sum(lengths)

        vectors = torch.cat(vectors, dim=0)
        outputs = self.act(self.linear1(vectors))
        logits = self.linear2(outputs)
        
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return logits, loss

        return logits

class MultitaskClassifierElectra(ElectraPreTrainedModel):
    def __init__(self, config, mlp_hid=16, use_contrastive_loss=0):
        super(MultitaskClassifierElectra, self).__init__(config)
        self.electra = ElectraModel(config)
        self.num_labels = 2
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear1 = nn.Linear(config.hidden_size, mlp_hid)
        self.linear2 = nn.Linear(mlp_hid, self.num_labels)
        self.act = nn.Tanh()
        self.init_weights()
        self.contrastive_loss = use_contrastive_loss

    def forward(self, input_ids, offsets, lengths, attention_mask=None,
                token_type_ids=None, position_ids=None, head_mask=None, labels=None):
        outputs = self.electra(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids)

        outputs = self.dropout(outputs[0])
                                                           
        idx = 0
        vectors = []
        for b, l in enumerate(lengths):
            for i in range(l):
                vectors.append(outputs[b, offsets[idx], :].unsqueeze(0))
                idx += 1
        assert idx == sum(lengths)

        vectors = torch.cat(vectors, dim=0)
        outputs = self.act(self.linear1(vectors))
        logits = self.linear2(outputs)
        
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return logits, loss

        return logits


class MultitaskClassifierRoberta(BertPreTrainedModel):
    #config_class = RobertaConfig
    #pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"
    def __init__(self, config, mlp_hid=16, use_parser_tag = False):
        super(MultitaskClassifierRoberta, self).__init__(config)
        self.roberta = RobertaModel(config)
        self.num_labels = 2
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear1 = nn.Linear(config.hidden_size, mlp_hid)
        self.linear2 = nn.Linear(mlp_hid, self.num_labels)
        self.act = nn.Tanh()

        self.use_parser_tag = use_parser_tag
        if self.use_parser_tag:
            self.linear_p = nn.Linear(config.hidden_size+len(depper_map)+len(tagger_map), config.hidden_size)
            self.gelu = nn.GELU()

        self.init_weights()

    def forward(self, input_ids, offsets, lengths, attention_mask=None,
                token_type_ids=None, position_ids=None, head_mask=None, labels=None,
                bpe_to_node=None, pos_labels = None, dep_labels = None):
        outputs = self.roberta(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               head_mask=head_mask)
        
        outputs = self.dropout(outputs[0])

        if self.use_parser_tag:
            new_sequence_output = torch.zeros((outputs.size(0), outputs.size(1), outputs.size(2)+len(tagger_map)+len(depper_map))).to(outputs.device)
            new_sequence_output[:,:,:outputs.size(-1)] = outputs
            #vectors = []
            for b, bpe_idxs in enumerate(bpe_to_node):
                pos_label = F.one_hot(torch.tensor(pos_labels[b]), num_classes = len(tagger_map)).to(outputs.device)
                dep_label = F.one_hot(torch.tensor(dep_labels[b]), num_classes = len(depper_map)).to(outputs.device)
                parser_label = torch.cat((pos_label, dep_label), dim = -1)
                for i, idx in enumerate(bpe_idxs):
                    new_sequence_output[b, idx, outputs.size(-1):].add_(parser_label[i,:])
            outputs = self.gelu(self.linear_p(new_sequence_output))

        idx = 0
        vectors = []
        for b, l in enumerate(lengths):
            for i in range(l):
                vectors.append(outputs[b, offsets[idx], :].unsqueeze(0))
                idx += 1
        assert idx == sum(lengths)

        vectors = torch.cat(vectors, dim=0)
        outputs = self.act(self.linear1(vectors))
        logits = self.linear2(outputs)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return logits, loss

        return logits


class MultitaskClassifierRoberta_ql(BertPreTrainedModel):
    config_class = RobertaConfig
    #pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"
    def __init__(self, config, mlp_hid=16, contrastive_loss_ratio=0, dropout_prob=0.2, question_loss_ratio=0, rank_loss_ratio = 0, len_q_labels = 0, num_labelss = 0):
        super(MultitaskClassifierRoberta_ql, self).__init__(config)
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(dropout_prob) if (dropout_prob >= 0 and dropout_prob<1) else nn.Dropout(config.hidden_dropout_prob)
        self.num_labels = num_labelss # ss for naming issue
        self.contrastive_loss_ratio = contrastive_loss_ratio
        self.question_loss_ratio = question_loss_ratio
        self.rank_loss_ratio = rank_loss_ratio

        if self.num_labels == 2:
            self.loss_fct = CrossEntropyLoss()
            self.softmax = nn.LogSoftmax(dim=0)
            if self.contrastive_loss_ratio:                
                self.loss_fct2 = NLLLoss()
            if self.rank_loss_ratio:
                self.event_rank_loss = NLLLoss()
        elif self.num_labels == 1:
            self.loss_fct = BCEWithLogitsLoss()
            self.softmax = nn.Softmax(dim=0)
            if self.contrastive_loss_ratio:
                self.loss_fct2 = BCELoss()
            if self.rank_loss_ratio:
                self.event_rank_loss = BCELoss()
        else:
            raise KeyError("num labels should be 1 or 2")

        self.linear1 = nn.Linear(config.hidden_size, mlp_hid)
        self.linear2 = nn.Linear(mlp_hid, self.num_labels)
        self.act = nn.Tanh()
        if question_loss_ratio: 
            self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
            self.predictor = nn.Linear(config.hidden_size, len_q_labels)
            self.loss_fct3 = nn.BCEWithLogitsLoss()

        self.init_weights()

    def forward(self, input_ids, offsets, lengths, attention_mask=None,
                token_type_ids=None, position_ids=None, head_mask=None, labels=None,
                 question_labels=None, question_mask=None, same_p_mask = None, return_all_logits=False):
        outputs = self.roberta(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               head_mask=head_mask)
        
        outputs = self.dropout(outputs[0])

        idx = 0
        vectors = []
        for b, l in enumerate(lengths):
            for i in range(l):
                vectors.append(outputs[b, offsets[idx], :].unsqueeze(0))
                idx += 1
        assert idx == sum(lengths)

        vectors = torch.cat(vectors, dim=0)

        if same_p_mask:
            orig_instance_ids = [qid[0] for qid in same_p_mask]

        if self.question_loss_ratio:
            # select question
            question_mask = question_mask.unsqueeze(-1).expand(question_mask.size(0), question_mask.size(1), outputs.size(-1))
            if same_p_mask:
                question_part = (outputs[orig_instance_ids] * question_mask).sum(dim=1) / question_mask.sum(dim=1)
            else: 
                question_part = (outputs * question_mask).sum(dim=1) / question_mask.sum(dim=1)
            question_part = self.act(self.pooler(question_part))
            question_logits = self.predictor(question_part) 
            question_loss = self.loss_fct3(question_logits, question_labels)

        outputs = self.act(self.linear1(vectors))
        logits = self.linear2(outputs)
        
        # original batch logits, labels
        if self.contrastive_loss_ratio and same_p_mask:
            lengths_sum = [range(sum(lengths[:i]), sum(lengths[:i+1])) for i in range(len(lengths))]
            orig_instance_idxs = [lengths_sum[inst_id] for inst_id in orig_instance_ids]
            orig_logits = logits[list(chain(*orig_instance_idxs))]
            orig_labels = labels[list(chain(*orig_instance_idxs))]
        else:
            orig_logits = logits
            orig_labels = labels

        if labels is not None:

            if self.num_labels == 2:
                loss = self.loss_fct(orig_logits.view(-1, self.num_labels), orig_labels.view(-1))
            elif self.num_labels == 1:
                loss = self.loss_fct(orig_logits.view(-1), orig_labels.view(-1).float())
            if self.contrastive_loss_ratio and same_p_mask:
                #softmax = nn.LogSoftmax(dim=0)
                contrastive_loss = 0
                for qids in same_p_mask:
                    neg_idxs = [lengths_sum[inst_id] for inst_id in qids]
                    neg_logits = logits[list(chain(*neg_idxs))]
                    neg_labels = labels[list(chain(*neg_idxs))]
                    temp_prob = self.softmax(neg_logits.reshape(len(qids), lengths[qids[0]], self.num_labels))
                    if self.num_labels == 1:
                        contrastive_loss += self.loss_fct2(temp_prob.view(-1), neg_labels.view(-1))
                    else:
                        contrastive_loss += self.loss_fct2(temp_prob.view(-1, self.num_labels), neg_labels.view(-1))
                contrastive_loss = contrastive_loss / len(same_p_mask)
                loss += self.contrastive_loss_ratio * contrastive_loss
            if self.rank_loss_ratio:
                #softmax = nn.Softmax(dim=0)
                start_length = 0
                event_rank_prob = torch.zeros_like(orig_logits, device=orig_logits.device)
                for l in lengths:
                    event_rank_prob[start_length:start_length+l] = self.softmax(orig_logits[start_length:start_length+l]) #\
                         #* (torch.sum(labels[start_length:start_length+l]).item())
                    start_length += l
                if self.num_labels == 1:
                    event_rank_loss = self.event_rank_loss(event_rank_prob.view(-1), orig_labels.view(-1).float())
                else:
                    event_rank_loss = self.event_rank_loss(event_rank_prob.view(-1, self.num_labels), orig_labels.view(-1))
                loss += self.rank_loss_ratio * event_rank_loss
            if self.question_loss_ratio:
                loss += self.question_loss_ratio * question_loss
            # if return_all_logits:
            #         return question_logits, logits, loss
            if self.num_labels == 1:
                orig_logits = torch.stack((1-torch.sigmoid(orig_logits), torch.sigmoid(orig_logits)), dim = -1).squeeze(1)
            return orig_logits, loss
        if self.num_labels == 1:
                orig_logits = torch.stack((1-torch.sigmoid(orig_logits), torch.sigmoid(orig_logits)), dim = -1).squeeze(1)
        return orig_logits

class MultitaskClassifierBart_ql(BartPretrainedModel):
    config_class = BartConfig
    #pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "bart"
    def __init__(self, config, mlp_hid=16, contrastive_loss_ratio=0, dropout_prob=0.2, question_loss_ratio=0, len_q_labels = 0):
        super(MultitaskClassifierBart_ql, self).__init__(config)
        self.bart = BartModel(config)
        self.dropout = nn.Dropout(dropout_prob) if dropout_prob >= 0 else nn.Dropout(config.dropout)
        self.num_labels = 2
        self.linear1 = nn.Linear(config.hidden_size, mlp_hid)
        self.linear2 = nn.Linear(mlp_hid, self.num_labels)
        self.linear1_encoder = nn.Linear(config.hidden_size, mlp_hid)
        self.linear2_encoder = nn.Linear(mlp_hid, self.num_labels)
        self.loss_fct = CrossEntropyLoss()
        self.act = nn.Tanh()
        if question_loss_ratio: 
            self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
            self.predictor = nn.Linear(config.hidden_size, len_q_labels)
            self.loss_fct3 = nn.BCEWithLogitsLoss()
        if contrastive_loss_ratio:
            self.loss_fct2 = NLLLoss()
        self.contrastive_loss_ratio = contrastive_loss_ratio
        self.question_loss_ratio = question_loss_ratio
        self.init_weights()

    def forward(self, input_ids, offsets, lengths, attention_mask=None,
                token_type_ids=None, position_ids=None, head_mask=None, labels=None,
                 question_labels=None, question_mask=None, same_p_mask = None, return_all_logits=False):
        outputs = self.bart(input_ids,
                               attention_mask=attention_mask,
                               #token_type_ids=token_type_ids,
                               #position_ids=position_ids,
                               head_mask=head_mask)
        
        decoder_outputs = self.dropout(outputs[0])
        encoder_outputs = self.dropout(outputs['encoder_last_hidden_state'])

        idx = 0
        decoder_vectors = []
        for b, l in enumerate(lengths):
            for i in range(l):
                decoder_vectors.append(decoder_outputs[b, offsets[idx], :].unsqueeze(0))
                idx += 1
        assert idx == sum(lengths)

        decoder_vectors = torch.cat(decoder_vectors, dim=0)

        idx = 0
        encoder_vectors = []
        for b, l in enumerate(lengths):
            for i in range(l):
                encoder_vectors.append(encoder_outputs[b, offsets[idx], :].unsqueeze(0))
                idx += 1
        assert idx == sum(lengths)

        encoder_vectors = torch.cat(encoder_vectors, dim=0)

        if same_p_mask:
            orig_instance_ids = [qid[0] for qid in same_p_mask]

        if self.question_loss_ratio:
            # select question
            question_mask = question_mask.unsqueeze(-1).expand(question_mask.size(0), question_mask.size(1), decoder_outputs.size(-1))
            if same_p_mask:
                question_part = (decoder_outputs[orig_instance_ids] * question_mask).sum(dim=1) / question_mask.sum(dim=1)
            else: 
                question_part = (decoder_outputs * question_mask).sum(dim=1) / question_mask.sum(dim=1)
            question_part = self.act(self.pooler(question_part))
            question_logits = self.predictor(question_part) 
            question_loss = self.loss_fct3(question_logits, question_labels)

        decoder_outputs = self.act(self.linear1(decoder_vectors))
        logits = self.linear2(decoder_outputs)
        
        encoder_outputs = self.act(self.linear1_encoder(encoder_vectors))
        encoder_logits = self.linear2_encoder(encoder_outputs)

        # original batch logits, labels
        if self.contrastive_loss_ratio and same_p_mask:
            lengths_sum = [range(sum(lengths[:i]), sum(lengths[:i+1])) for i in range(len(lengths))]
            orig_instance_idxs = [lengths_sum[inst_id] for inst_id in orig_instance_ids]
            orig_logits = logits[list(chain(*orig_instance_idxs))]
            orig_labels = labels[list(chain(*orig_instance_idxs))]
        else:
            orig_logits = logits
            orig_labels = labels

        if labels is not None:
            loss = self.loss_fct(orig_logits.view(-1, self.num_labels), orig_labels.view(-1)) + self.loss_fct(encoder_logits.view(-1, self.num_labels), orig_labels.view(-1))
            if self.contrastive_loss_ratio and same_p_mask:
                softmax = nn.LogSoftmax(dim=0)
                contrastive_loss = 0
                for qids in same_p_mask:
                    neg_idxs = [lengths_sum[inst_id] for inst_id in qids]
                    neg_logits = logits[list(chain(*neg_idxs))]
                    neg_labels = labels[list(chain(*neg_idxs))]
                    temp_prob = softmax(neg_logits.reshape(len(qids), lengths[qids[0]], self.num_labels))
                    contrastive_loss += self.loss_fct2(temp_prob.view(-1, self.num_labels), neg_labels.view(-1))
                contrastive_loss = contrastive_loss / len(same_p_mask)
                loss += self.contrastive_loss_ratio * contrastive_loss
            if self.question_loss_ratio:
                loss += self.question_loss_ratio * question_loss
            return logits, loss

        return logits

class MultitaskClassifierRoberta_OTR(BertPreTrainedModel):
    config_class = RobertaConfig
    #pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"
    def __init__(self, config, batch_size, mlp_hid=16, contrastive_loss_ratio=0, dropout_prob=0.2, question_loss_ratio=0, len_q_labels=0):
        super(MultitaskClassifierRoberta_OTR, self).__init__(config)
        self.contrastive_loss_ratio = contrastive_loss_ratio
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(dropout_prob) if dropout_prob >= 0 else nn.Dropout(config.hidden_dropout_prob)
        self.num_labels = 2
        self.linear1 = nn.Linear(config.hidden_size, mlp_hid)
        self.linear2 = nn.Linear(mlp_hid, self.num_labels)
        self.loss_fct = CrossEntropyLoss()
        self.loss_fct2 = NLLLoss()
        self.act = nn.Tanh()
        if question_loss_ratio: 
            # prev one 
            self.pooler = nn.Linear(config.hidden_size, len_q_labels)
            self.loss_fct3 = nn.BCEWithLogitsLoss()
            self.linear3 = nn.Linear(config.hidden_size+len_q_labels, config.hidden_size)
        self.contrastive_loss_ratio = contrastive_loss_ratio
        self.question_loss_ratio = question_loss_ratio
        self.batch_size = batch_size
        self.init_weights()

    def forward(self, input_ids, offsets, lengths, attention_mask=None,
                token_type_ids=None, position_ids=None, head_mask=None, labels=None, question_labels=None, question_mask=None, same_p_mask=None):
        outputs = self.roberta(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               head_mask=head_mask)
        
        outputs = self.dropout(outputs[0])

        idx = 0
        vectors = []
        for b, l in enumerate(lengths):
            for i in range(l):
                vectors.append(outputs[b, offsets[idx], :].unsqueeze(0))
                idx += 1
        assert idx == sum(lengths)

        vectors = torch.cat(vectors, dim=0)

        if self.question_loss_ratio:
            # select question
            #question_part = torch.zeros(outputs.size()).to(input_ids.device)
            #question_part[question_mask] = outputs[question_mask]
            question_mask = question_mask.unsqueeze(-1).expand(question_mask.size(0), question_mask.size(1), outputs.size(-1))
            question_part = (outputs * question_mask).sum(dim=1)/question_mask.sum(dim=1)
            # question_part = self.act(self.pooler(question_part))
            # question_logits = self.predictor(question_part) 
            # question_loss = self.loss_fct3(question_logits, question_labels)
            
            question_logits = self.act(self.pooler(question_part))
            question_loss = self.loss_fct3(question_logits, question_labels)

            vectors = vectors.view(input_ids.size(0), -1, vectors.size(-1))
            #question_part = self.dropout(question_part)
            #question_part = question_part.unsqueeze(1).expand(question_part.size(0),vectors.size(1),question_part.size(-1))
            #vectors = self.linear3(torch.cat([vectors, question_part], dim=-1))
            question_logits = self.dropout(question_logits)
            question_logits = question_logits.unsqueeze(1).expand(question_logits.size(0),vectors.size(1),question_logits.size(-1))
            #vectors = self.linear3(torch.cat([vectors, question_part], dim=-1))            
            vectors = self.linear3(torch.cat([vectors, question_logits], dim=-1))            
            vectors = vectors.view(-1, outputs.size(-1))


        outputs = self.act(self.linear1(vectors))
        logits = self.linear2(outputs)

        if labels is not None:
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            if self.contrastive_loss_ratio:
                softmax = nn.LogSoftmax(dim=0)
                contrast_prob = softmax(logits.view(input_ids.size(0), -1, self.num_labels))
                contrastive_loss = self.loss_fct2(contrast_prob.view(-1, self.num_labels), labels.view(-1))
                loss += self.contrastive_loss_ratio * contrastive_loss 
            if self.question_loss_ratio:
                loss += self.question_loss_ratio * question_loss
            return logits, loss

        return logits

# merged with ql
class MultitaskClassifierRoberta_OTR2(BertPreTrainedModel):
    pass
#     #config_class = RobertaConfig
#     #pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
#     #base_model_prefix = "roberta"
#     def __init__(self, config, mlp_hid=16, contrastive_loss_ratio=0, dropout_prob=0.2):
#         super(MultitaskClassifierRoberta_OTR2, self).__init__(config)
#         self.contrastive_loss_ratio = contrastive_loss_ratio
#         self.roberta = RobertaModel(config)
#         self.dropout = nn.Dropout(dropout_prob)
#         self.num_labels = 2
#         self.linear1 = nn.Linear(config.hidden_size, mlp_hid)
#         self.linear2 = nn.Linear(mlp_hid, self.num_labels)
#         self.loss_fct = CrossEntropyLoss()
#         self.loss_fct2 = NLLLoss()
#         self.act = nn.Tanh()
#         self.init_weights()
#         self.contrastive_loss_ratio = contrastive_loss_ratio
        

#     def forward(self, input_ids, offsets, lengths, attention_mask=None,
#                 token_type_ids=None, position_ids=None, head_mask=None, labels=None, same_p_mask = None):
#         outputs = self.roberta(input_ids,
#                                attention_mask=attention_mask,
#                                token_type_ids=token_type_ids,
#                                position_ids=position_ids,
#                                head_mask=head_mask)
        
#         outputs = self.dropout(outputs[0])

#         idx = 0
#         vectors = []
#         for b, l in enumerate(lengths):
#             for i in range(l):
#                 vectors.append(outputs[b, offsets[idx], :].unsqueeze(0))
#                 idx += 1
#         assert idx == sum(lengths)

#         vectors = torch.cat(vectors, dim=0)
        
#         outputs = self.act(self.linear1(vectors))
#         logits = self.linear2(outputs)

#         lengths_sum = [range(sum(lengths[:i]), sum(lengths[:i+1])) for i in range(len(lengths))]
        
#         # original batch logits, labels
#         if same_p_mask is not None:
#             orig_instance_ids = [qid[0] for qid in same_p_mask]
#             orig_instance_idxs = [lengths_sum[inst_id] for inst_id in orig_instance_ids]
#             orig_logits = logits[list(chain(*orig_instance_idxs))]
#             orig_labels = labels[list(chain(*orig_instance_idxs))]
#         else:
#             orig_logits = logits
#             orig_labels = labels
#         # negative q labels (for each instance in batch)

#         if labels is not None:
#             loss = self.loss_fct(orig_logits.view(-1, self.num_labels), orig_labels.view(-1))
#             if self.contrastive_loss_ratio and same_p_mask:
#                 softmax = nn.LogSoftmax(dim=0)
#                 contrastive_loss = 0
#                 for qids in same_p_mask:
#                     neg_idxs = [lengths_sum[inst_id] for inst_id in qids]
#                     neg_logits = logits[list(chain(*neg_idxs))]
#                     neg_labels = labels[list(chain(*neg_idxs))]
#                     temp_prob = softmax(neg_logits.reshape(len(qids), lengths[qids[0]], self.num_labels))
#                     contrastive_loss += self.loss_fct2(temp_prob.view(-1, self.num_labels), neg_labels.view(-1))
#                 contrastive_loss = contrastive_loss / len(same_p_mask)
#                 loss += self.contrastive_loss_ratio * contrastive_loss
#             return logits, loss 

#         return logits


"""
TODO: meerge with ql1
calculate contrast

"""
class MultitaskClassifierRoberta_ql2(BertPreTrainedModel):
    config_class = RobertaConfig
    #pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"
    def __init__(self, config, mlp_hid=16, contrastive_loss_ratio=0, dropout_prob=0.2, question_loss_ratio=0, rank_loss_ratio = 0, len_q_labels = 0, num_labelss = 0):
        super(MultitaskClassifierRoberta_ql, self).__init__(config)
        self.roberta = RobertaModel(config)
        self.lm_head = RobertaLMHead(config)
        self.dropout = nn.Dropout(dropout_prob) if (dropout_prob >= 0 and dropout_prob<1) else nn.Dropout(config.hidden_dropout_prob)
        self.num_labels = num_labelss # ss for naming issue
        self.contrastive_loss_ratio = contrastive_loss_ratio
        self.question_loss_ratio = question_loss_ratio
        self.rank_loss_ratio = rank_loss_ratio

        if self.num_labels == 2:
            self.loss_fct = CrossEntropyLoss()
            self.softmax = nn.LogSoftmax(dim=0)
            if self.contrastive_loss_ratio:                
                self.loss_fct2 = NLLLoss()
            if self.rank_loss_ratio:
                self.event_rank_loss = NLLLoss()
        elif self.num_labels == 1:
            self.loss_fct = BCEWithLogitsLoss()
            self.softmax = nn.Softmax(dim=0)
            if self.contrastive_loss_ratio:
                self.loss_fct2 = BCELoss()
            if self.rank_loss_ratio:
                self.event_rank_loss = BCELoss()
        else:
            raise KeyError("num labels should be 1 or 2")

        self.linear1 = nn.Linear(config.hidden_size, mlp_hid)
        self.linear2 = nn.Linear(mlp_hid, self.num_labels)
        self.act = nn.Tanh()
        if question_loss_ratio: 
            self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
            self.predictor = nn.Linear(config.hidden_size, len_q_labels)
            self.loss_fct3 = nn.BCEWithLogitsLoss()

        self.init_weights()

    def forward(self, input_ids, offsets, lengths, attention_mask=None,
                token_type_ids=None, position_ids=None, head_mask=None, labels=None,
                 question_labels=None, question_mask=None, same_p_mask = None, events= None, return_all_logits=False):
        outputs = self.roberta(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               head_mask=head_mask)
        
        outputs = self.dropout(outputs[0])
        
        #50264 = roberta-large <mask> token id
        mask_indexs = (input_ids == 50264).nonzero()
        if len(mask_indexs):
            pass
        idx = 0
        vectors = []
        for b, l in enumerate(lengths):
            for i in range(l):
                vectors.append(outputs[b, offsets[idx], :].unsqueeze(0))
                idx += 1
        assert idx == sum(lengths)

        vectors = torch.cat(vectors, dim=0)

        if same_p_mask:
            orig_instance_ids = [qid[0] for qid in same_p_mask]

        if self.question_loss_ratio:
            # select question
            question_mask = question_mask.unsqueeze(-1).expand(question_mask.size(0), question_mask.size(1), outputs.size(-1))
            if same_p_mask:
                question_part = (outputs[orig_instance_ids] * question_mask).sum(dim=1) / question_mask.sum(dim=1)
            else: 
                question_part = (outputs * question_mask).sum(dim=1) / question_mask.sum(dim=1)
            question_part = self.act(self.pooler(question_part))
            question_logits = self.predictor(question_part) 
            question_loss = self.loss_fct3(question_logits, question_labels)

        outputs = self.act(self.linear1(vectors))
        logits = self.linear2(outputs)
        
        # original batch logits, labels
        if self.contrastive_loss_ratio and same_p_mask:
            lengths_sum = [range(sum(lengths[:i]), sum(lengths[:i+1])) for i in range(len(lengths))]
            orig_instance_idxs = [lengths_sum[inst_id] for inst_id in orig_instance_ids]
            orig_logits = logits[list(chain(*orig_instance_idxs))]
            orig_labels = labels[list(chain(*orig_instance_idxs))]
        else:
            orig_logits = logits
            orig_labels = labels

        if labels is not None:

            if self.num_labels == 2:
                loss = self.loss_fct(orig_logits.view(-1, self.num_labels), orig_labels.view(-1))
            elif self.num_labels == 1:
                loss = self.loss_fct(orig_logits.view(-1), orig_labels.view(-1).float())
            if self.contrastive_loss_ratio and same_p_mask:
                #softmax = nn.LogSoftmax(dim=0)
                contrastive_loss = 0
                for qids in same_p_mask:
                    neg_idxs = [lengths_sum[inst_id] for inst_id in qids]
                    neg_logits = logits[list(chain(*neg_idxs))]
                    neg_labels = labels[list(chain(*neg_idxs))]
                    temp_prob = self.softmax(neg_logits.reshape(len(qids), lengths[qids[0]], self.num_labels))
                    if self.num_labels == 1:
                        contrastive_loss += self.loss_fct2(temp_prob.view(-1), neg_labels.view(-1))
                    else:
                        contrastive_loss += self.loss_fct2(temp_prob.view(-1, self.num_labels), neg_labels.view(-1))
                contrastive_loss = contrastive_loss / len(same_p_mask)
                loss += self.contrastive_loss_ratio * contrastive_loss
            if self.rank_loss_ratio:
                #softmax = nn.Softmax(dim=0)
                start_length = 0
                event_rank_prob = torch.zeros_like(orig_logits, device=orig_logits.device)
                for l in lengths:
                    event_rank_prob[start_length:start_length+l] = self.softmax(orig_logits[start_length:start_length+l]) #\
                         #* (torch.sum(labels[start_length:start_length+l]).item())
                    start_length += l
                if self.num_labels == 1:
                    event_rank_loss = self.event_rank_loss(event_rank_prob.view(-1), orig_labels.view(-1).float())
                else:
                    event_rank_loss = self.event_rank_loss(event_rank_prob.view(-1, self.num_labels), orig_labels.view(-1))
                loss += self.rank_loss_ratio * event_rank_loss
            if self.question_loss_ratio:
                loss += self.question_loss_ratio * question_loss
            # if return_all_logits:
            #         return question_logits, logits, loss
            if self.num_labels == 1:
                orig_logits = torch.stack((1-torch.sigmoid(orig_logits), torch.sigmoid(orig_logits)), dim = -1).squeeze(1)
            return orig_logits, loss
        if self.num_labels == 1:
                orig_logits = torch.stack((1-torch.sigmoid(orig_logits), torch.sigmoid(orig_logits)), dim = -1).squeeze(1)
        return orig_logits