from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import BertPreTrainedModel, BertModel, BertConfig, RobertaConfig, RobertaModel
from gnns import GCN
from models_encoderlayer import TransformerEncoderLayerWithCrossAttention
from models_graph import ResidualGRU

logger = logging.getLogger(__name__)

BERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-pytorch_model.bin",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-pytorch_model.bin",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-pytorch_model.bin",
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-pytorch_model.bin",
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-pytorch_model.bin",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-pytorch_model.bin",
    'bert-base-german-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-cased-pytorch_model.bin",
    'bert-large-uncased-whole-word-masking': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-pytorch_model.bin",
    'bert-large-cased-whole-word-masking': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-pytorch_model.bin",
    'bert-large-uncased-whole-word-masking-finetuned-squad': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-pytorch_model.bin",
    'bert-large-cased-whole-word-masking-finetuned-squad': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-finetuned-squad-pytorch_model.bin",
    'bert-base-cased-finetuned-mrpc': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-finetuned-mrpc-pytorch_model.bin",
}

ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'roberta-base': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-pytorch_model.bin",
    'roberta-large': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-pytorch_model.bin",
    'roberta-large-mnli': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-pytorch_m\
odel.bin",
}

class TEClassifierRoberta(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"
    def __init__(self, config, mlp_hid=16, num_classes=5, finetune=True):
        super(TEClassifierRoberta, self).__init__(config)
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.act = nn.Tanh()
        self.num_classes = num_classes
        self.linear1_te = nn.Linear(config.hidden_size*2, mlp_hid)
        self.linear2_te = nn.Linear(mlp_hid, self.num_classes)

        self.init_weights()
        if not finetune:
            for name, param in self.roberta.named_parameters():
                param.requires_grad = False

    def forward(self, input_ids_te, token_type_ids_te=None, attention_mask_te=None,
                lidx_s=None, lidx_e=None, ridx_s=None, ridx_e=None, length_te=None,
                labels_te=None):

        batch_max = length_te.max()
        flat_input_ids_te = input_ids_te#.view(-1, input_ids_te.size(-1))[:, :batch_max]
        flat_token_type_ids_te = token_type_ids_te#.view(-1, token_type_ids_te.size(-1))[:, :batch_max]
        flat_attention_mask_te = attention_mask_te#.view(-1, attention_mask_te.size(-1))[:, :batch_max]
        output = self.roberta(flat_input_ids_te,
                                       token_type_ids=flat_token_type_ids_te,
                                       attention_mask=flat_attention_mask_te)
        out = output[0]
        batch = out.size(0)

        ltar_s = torch.cat([out[b, lidx_s[b], :] for b in range(batch)], dim=0).squeeze(1)
        rtar_s = torch.cat([out[b, ridx_s[b], :] for b in range(batch)], dim=0).squeeze(1)
        out = self.dropout(torch.cat([ltar_s, rtar_s], dim=1))
        
        # print(out.size())
        # linear prediction
        out = self.linear1_te(out)
        # print(out.size())
        out = self.act(out)
        out = self.linear2_te(out)
        # print(out.size())
        logits_te = out.view(-1, self.num_classes)


        if self.deliberate:
            out2 = self._cont_shuffler(out)        



        if labels_te is not None:
            loss_fct = CrossEntropyLoss()
            loss_te = loss_fct(logits_te, labels_te)
            return loss_te, logits_te
        else:
            return logits_te



class TEClassifierRobertaGraph(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config, mlp_hid=16, num_classes=5, gcn_steps=3, max_question_length=4,
                 use_gcn=False, event_loss_ratio=0.0, use_parser_tag=False,
                 contrastive_loss_ratio=0, dropout_prob=-1, rank_loss_ratio=0,
                 wo_events=False, residual_connection=False, question_concat=False,
                 deliberate=0, ablation=0, share=False, deliberate_ffn=2048):
        super(TEClassifierRobertaGraph, self).__init__(config)
        self.roberta = RobertaModel(config)
        #self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.act = nn.Tanh()
        self.num_classes = num_classes
        self.linear1_te = nn.Linear(config.hidden_size*2, mlp_hid)
        self.linear2_te = nn.Linear(mlp_hid, self.num_classes)

        #added
        dropout_prob = dropout_prob if dropout_prob >=0 else config.hidden_dropout_prob
        self.dropout = nn.Dropout(dropout_prob)
        self.linear1 = nn.Linear(config.hidden_size, mlp_hid)
        self.linear2 = nn.Linear(mlp_hid, self.num_classes)
        self.act = nn.Tanh()
        
        self.use_gcn = use_gcn
        self.residual_connection = residual_connection
        if self.use_gcn:
            node_dim = config.hidden_size
            self._gcn_input_proj = nn.Linear(node_dim * 4, node_dim)
            self._gcn = GCN(node_dim=node_dim, iteration_steps=gcn_steps)
            #self._iteration_steps = gcn_steps
            self._proj_ln = nn.LayerNorm(node_dim)
            self._gcn_enc = ResidualGRU(config.hidden_size, config.attention_probs_dropout_prob, 2)
            self.max_question_length = max_question_length

        self.deliberate = deliberate
        if self.deliberate:
            encoder_layer = TransformerEncoderLayerWithCrossAttention(config, dim_feedforward=deliberate_ffn, ablation=ablation)
            norm = nn.LayerNorm(config.hidden_size)
            self._cont_shuffler = nn.TransformerEncoder(encoder_layer, self.deliberate, norm)

        self.linear3_te = nn.Linear(config.hidden_size, mlp_hid)
        self.linear4_te = nn.Linear(mlp_hid, self.num_classes)

        self.init_weights()
        # if not finetune:
        #     for name, param in self.roberta.named_parameters():
        #         param.requires_grad = False

    def forward(self, input_ids_te, lengths, offsets, token_type_ids_te=None, attention_mask_te=None,
                lidx_s=None, lidx_e=None, ridx_s=None, ridx_e=None, length_te=None,
                labels_te=None,
                bpe_to_node=None, edges=None, question_len=None,):

        # batch_max = length_te.max()
        flat_input_ids_te = input_ids_te#.view(-1, input_ids_te.size(-1))[:, :batch_max]
        flat_token_type_ids_te = token_type_ids_te#.view(-1, token_type_ids_te.size(-1))[:, :batch_max]
        flat_attention_mask_te = attention_mask_te#.view(-1, attention_mask_te.size(-1))[:, :batch_max]
        outputs = self.roberta(flat_input_ids_te,
                                       token_type_ids=flat_token_type_ids_te,
                                       attention_mask=flat_attention_mask_te, output_hidden_states=True)
        sequence_output = outputs[0]

        if self.use_gcn:
            sequence_output_list = [item for item in outputs[2][-4:] ] # last 4 hidden states
            sequence_alg = self._gcn_input_proj(torch.cat([layer_output for layer_output in sequence_output_list], dim=2))
            
            total_word_length = [len(h) for h in bpe_to_node]
            doc_len = [total_word_length[i]-question_len[i] for i in range(len(total_word_length))]
            #max_word_length = max(total_word_length).item()
            max_p_word_length = max(doc_len)
            max_q_word_length = max(question_len)

            qq_graph, dq_graph, dd_graph, qd_graph = self.draw_graphs(edges, max_p_word_length, max_q_word_length)
            
            encoded_doc = self.select_encoded_doc(sequence_alg, doc_len, question_len, bpe_to_node)
            encoded_question = self.select_encoded_question(sequence_alg, question_len, bpe_to_node)

            d_node, _, _, _ = self._gcn(d_node=encoded_doc, q_node=encoded_question,
                    qq_graph=qq_graph, dq_graph=dq_graph, dd_graph = dd_graph, qd_graph = qd_graph)

            for b, l in enumerate(bpe_to_node):
                for i, pos in enumerate(l[question_len[b]:]):
                    sequence_output[b, pos].add_(d_node[b, i])
            sequence_output = self._proj_ln(sequence_output)

        batch_size = sequence_output.size(0)

        idx = 0
        vectors = []
        #lidx_ss = []
        #ridx_ss = []
        for b, l in enumerate(lengths):
            #lidx_ss.append(offsets[idx + lidx_s[b]])
            #ridx_ss.append(offsets[idx + ridx_s[b]])
            for i in range(l):
                vectors.append(sequence_output[b, offsets[idx], :].unsqueeze(0)) 
                idx += 1
        assert idx == sum(lengths)
        vectors = torch.cat(vectors, dim=0)
        vectors = vectors.reshape(batch_size, -1, vectors.size(-1))

        # ltar_s = torch.cat([vectors[b, lidx_s[b], :] for b in range(batch_size)], dim=0).squeeze(1)
        # rtar_s = torch.cat([vectors[b, ridx_s[b], :] for b in range(batch_size)], dim=0).squeeze(1)
        ltar_s = torch.stack([vectors[b, lidx_s[b], :] for b in range(batch_size)], dim=0)
        rtar_s = torch.stack([vectors[b, ridx_s[b], :] for b in range(batch_size)], dim=0)
        out_temp = self.dropout(torch.cat([ltar_s, rtar_s], dim=1))
        # linear prediction
        out = self.linear1_te(out_temp)
        out = self.act(out)
        out = self.linear2_te(out)
        logits_te = out.view(-1, self.num_classes)

        if self.deliberate:
            vectors2 = self._cont_shuffler(vectors)

            ltar_s = torch.stack([vectors2[b, lidx_s[b], :] for b in range(batch_size)], dim=0)
            rtar_s = torch.stack([vectors2[b, ridx_s[b], :] for b in range(batch_size)], dim=0)
            out2 = self.dropout(torch.cat([ltar_s, rtar_s], dim=1))

            if self.residual_connection:
                out2 = out2 + out_temp
            # linear prediction
            out2 = self.linear1_te(out2)
            out2 = self.act(out2)
            out2 = self.linear2_te(out2)
            logits_te2 = out2.view(-1, self.num_classes)

        if labels_te is not None:
            loss_fct = CrossEntropyLoss()
            if self.deliberate:
                loss_te = loss_fct(logits_te, labels_te) + loss_fct(logits_te2, labels_te)
                loss_te = loss_te/2
                logits_te = logits_te2
            else:
                loss_te = loss_fct(logits_te, labels_te)
            return loss_te, logits_te
        else:
            return logits_te

    def select_encoded_doc(self, sequence_alg, doc_len, question_len, bpe_to_node):
        batch_size, _, hidden_size = sequence_alg.size()
        encoded_doc = torch.zeros(batch_size, max(doc_len), hidden_size).to(sequence_alg.device)
        for batch_id in range(batch_size):
            for doc_idx, bpe_idx in enumerate(bpe_to_node[batch_id][question_len[batch_id]:]):
                encoded_doc[batch_id][doc_idx] = sequence_alg[batch_id][bpe_idx]
        return encoded_doc
    
    def select_encoded_question(self, sequence_alg, question_len, bpe_to_node):
        batch_size, _, hidden_size = sequence_alg.size()
        encoded_question = torch.zeros(batch_size, max(question_len), hidden_size).to(sequence_alg.device)
        for batch_id in range(batch_size):
            for doc_idx, bpe_idx in enumerate(bpe_to_node[batch_id][:question_len[batch_id]]):
                encoded_question[batch_id][doc_idx] = sequence_alg[batch_id][bpe_idx]
        return encoded_question

    def draw_graphs(self, edges, max_p_word_length, max_q_word_length):
        batch_size = edges.size(0)
        graph = torch.zeros(batch_size, max_p_word_length + self.max_question_length, max_p_word_length + self.max_question_length).to(edges.device)
        for batch_id in range(batch_size):    
            graph[batch_id][edges[batch_id,:,0], edges[batch_id,:,1]] = 1
            #graph[batch_id][edges[batch_id,:,1], edges[batch_id,:,0]] = 1

        qq_graph = graph[:, :max_q_word_length, :max_q_word_length]
        dd_graph = graph[:, self.max_question_length:, self.max_question_length:]
        qd_graph = graph[:, :max_q_word_length, self.max_question_length:]
        #dq_graph=qd_graph.reshape(-1, qd_graph.size(2), qd_graph.size(1))
        dq_graph=qd_graph.transpose(1,2)
        
        return qq_graph, dq_graph, dd_graph, qd_graph