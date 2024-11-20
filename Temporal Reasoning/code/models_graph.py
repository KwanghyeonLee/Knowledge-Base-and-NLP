import torch
from torch import nn
import torch.nn.functional as F
from utils import replace_masked_values, depper_map, tagger_map
from torch.nn import CrossEntropyLoss, NLLLoss, BCEWithLogitsLoss, BCELoss
from transformers import BertPreTrainedModel, BertModel, RobertaModel, RobertaConfig, \
    BertConfig, DebertaConfig, DebertaModel, DebertaPreTrainedModel, \
    DebertaV2Config, DebertaV2Model, DebertaV2PreTrainedModel, \
    ElectraConfig, ElectraModel, ElectraPreTrainedModel
from transformers.models.roberta.modeling_roberta import RobertaLMHead
from gnns import GCN
from itertools import chain
from models_encoderlayer import TransformerEncoderLayerWithCrossAttention

class NumNetMultitaskClassifierRoberta(BertPreTrainedModel):
    config_class = RobertaConfig
    def __init__(self, config, mlp_hid=16, gcn_steps=4, max_question_length=50, use_gcn=False, event_loss_ratio=0.0, use_parser_tag=False,
    contrastive_loss_ratio=0, dropout_prob = -1, rank_loss_ratio=0, num_labelss=2, wo_events=False, question_concat = False):
        super(NumNetMultitaskClassifierRoberta, self).__init__(config)
        self.roberta = RobertaModel(config)
        self.num_labels = num_labelss
        self.contrastive_loss_ratio = contrastive_loss_ratio
        self.rank_loss_ratio = rank_loss_ratio
        self.wo_events = wo_events
        self.question_concat = question_concat
        if question_concat:
            self.concatlayer = nn.Linear(config.hidden_size * 2, config.hidden_size)
        if self.num_labels == 2:
            self.loss_fct = CrossEntropyLoss()
            self.softmax = nn.LogSoftmax(dim=0)
            if self.contrastive_loss_ratio:                
                self.cont_loss = NLLLoss()
            if self.rank_loss_ratio:
                self.event_rank_loss = NLLLoss()
                self.lm_head = RobertaLMHead(config)
                self.event_projector = EventProjector(config.hidden_size)
        elif self.num_labels == 1:
            self.softmax = nn.Softmax(dim=0)
            self.loss_fct = BCEWithLogitsLoss()
            if self.contrastive_loss_ratio:
                self.cont_loss = BCELoss()
            # if self.rank_loss_ratio:
            #     self.event_rank_loss = BCELoss()
        else:
            raise KeyError("num labels should be 1 or 2")

        dropout_prob = dropout_prob if dropout_prob >=0 else config.hidden_dropout_prob
        self.dropout = nn.Dropout(dropout_prob)
        self.linear1 = nn.Linear(config.hidden_size, mlp_hid)
        self.linear2 = nn.Linear(mlp_hid, self.num_labels)
        self.act = nn.Tanh()

        self.event_loss_ratio=event_loss_ratio
        if self.rank_loss_ratio:
            self.gelu=nn.GELU()
            self.event_projector = EventProjector(config.hidden_size)
        if self.event_loss_ratio:
            self.event_detector = nn.Linear(config.hidden_size, self.num_labels)
        self.use_parser_tag=use_parser_tag
        if self.use_parser_tag:
            self.linear_p = nn.Linear(config.hidden_size+len(depper_map)+len(tagger_map), config.hidden_size)
            self.gelu = nn.GELU()        

        self.use_gcn=use_gcn
        if self.use_gcn:
            node_dim = config.hidden_size
            self._gcn_input_proj = nn.Linear(node_dim * 2, node_dim)
            self._gcn = GCN(node_dim=node_dim, iteration_steps=gcn_steps)
            #self._iteration_steps = gcn_steps
            self._proj_ln = nn.LayerNorm(node_dim)
            self._gcn_enc = ResidualGRU(config.hidden_size, config.attention_probs_dropout_prob, 2)
            self.max_question_length = max_question_length
        self.init_weights()

    def forward(self, input_ids, offsets, lengths, bpe_to_node, edges, question_len, attention_mask=None,
                token_type_ids=None, position_ids=None, head_mask=None, labels=None, events=None, 
                q_event_word_idx = None, q_tr_word_idx = None, parser_labels = None, same_p_mask=None):
        #print("1", datetime.now().time())
        outputs = self.roberta(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               head_mask=head_mask, output_hidden_states=True)

        sequence_output = self.dropout(outputs[0]) # last hidden states
        #print("2", datetime.now().time())
        if self.use_parser_tag:
            new_sequence_output = torch.zeros((sequence_output.size(0), sequence_output.size(1), sequence_output.size(2)+parser_labels.size(2))).to(sequence_output.device)
            new_sequence_output[:,:,:sequence_output.size(-1)] = sequence_output
            idx = 0
            #vectors = []
            for b, l in enumerate(lengths):
                for i in range(l):
                    #torch.cat(sequence_output[b, offsets[idx], :],parser_labels[b, i, :])
                    new_sequence_output[b, offsets[idx], sequence_output.size(-1):] = parser_labels[b, i, :]
                    idx += 1
            assert idx == sum(lengths)
            sequence_output = new_sequence_output
            sequence_output = self.gelu(self.linear_p(new_sequence_output))

        if self.use_gcn:
            sequence_output_list = [ item for item in outputs[2][-2:] ] # last 2 hidden states
            sequence_alg = self._gcn_input_proj(torch.cat([sequence_output_list[0], sequence_output_list[1]], dim=2))
            
            total_word_length = [len(h) for h in bpe_to_node]
            doc_len = [total_word_length[i]-question_len[i] for i in range(len(total_word_length))]
            #max_word_length = max(total_word_length).item()
            max_p_word_length = max(doc_len)
            max_q_word_length = max(question_len)
            #word_level_outputs = self.resize_outputs(sequence_alg, head_masks, max_word_length)

            qq_graph, dq_graph, dd_graph, qd_graph = self.draw_graphs(edges, max_p_word_length, max_q_word_length)
            
            encoded_doc = self.select_encoded_doc(sequence_alg, doc_len, question_len, bpe_to_node)
            encoded_question = self.select_encoded_question(sequence_alg, question_len, bpe_to_node)

            d_node, _, _, _ = self._gcn(d_node=encoded_doc, q_node=encoded_question,
                    qq_graph=qq_graph, dq_graph=dq_graph, dd_graph = dd_graph, qd_graph = qd_graph)

            for b, l in enumerate(bpe_to_node):
                for i, pos in enumerate(l[question_len[b]:]):
                    sequence_output[b, pos].add_(d_node[b, i])
                    #sequence_output[b, pos] = torch.add(sequence_output[b, pos] ,d_node[b, i])
       
            #sequence_output = self.dropout(self._gcn_enc(self._proj_ln(sequence_output)))
            sequence_output = self._proj_ln(sequence_output)

        if self.rank_loss_ratio:
            x = self.lm_head.dense(sequence_output)
            x = self.gelu(x)
            x = self.lm_head.layer_norm(x)
            #event_rank_loss = self.event_projector(new_input_ids, x, new_sequence_output, new_events, new_labels, offsets, lengths)
            event_rank_loss = self.event_projector(input_ids, x, sequence_output, events, labels, offsets, lengths)
            

        if self.question_concat:
            # select encoded question and mean
            idx = 0
            vectors = []
            for b, l in enumerate(lengths):
                encoded_question = sequence_output[b,1:bpe_to_node[b][question_len[b]]].mean(0)
                for i in range(l):
                    vectors.append(torch.cat([sequence_output[b, offsets[idx], :].unsqueeze(0), encoded_question.unsqueeze(0)], dim=-1))
                    idx += 1
            assert idx == sum(lengths)
            vectors = torch.cat(vectors, dim=0)
            vectors = self.concatlayer(vectors)

        else:
            idx = 0
            vectors = []
            for b, l in enumerate(lengths):
                for i in range(l):
                    vectors.append(sequence_output[b, offsets[idx], :].unsqueeze(0))
                    idx += 1
            assert idx == sum(lengths)
            vectors = torch.cat(vectors, dim=0)

        outputs = self.act(self.linear1(vectors))
        logits = self.linear2(outputs)
    
        if self.contrastive_loss_ratio and not self.wo_events: #FIXME: args.wo_events
            # set event label again. get the longest
            event_tmp = events[0:lengths[0]] # get first
            start_length=0
            for l in lengths:
                event_tmp = torch.logical_or(event_tmp, events[start_length:start_length+l])
                start_length+=l
            event_lens = [sum(event_tmp)] * len(lengths)
            event_tmp = event_tmp.repeat(len(lengths)) 
            events_mask = event_tmp.unsqueeze(-1).expand(-1,self.num_labels)
            event_logits = torch.masked_select(logits,events_mask.bool()).reshape(-1, self.num_labels)
            event_labels = torch.masked_select(labels, event_tmp.bool())    
        else:
            event_logits = logits
            event_labels = labels        
        if labels is not None:
            if self.num_labels == 2:
                loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.num_labels == 1:
                loss = self.loss_fct(logits.view(-1), labels.view(-1).float())
            if self.event_loss_ratio:
            #     # event label is only given during training
                event_pred = self.event_detector(vectors)
                event_fct = CrossEntropyLoss()
                event_loss = event_fct(event_pred.view(-1,self.num_labels), events.view(-1))
                loss += self.event_loss_ratio * event_loss
            if self.contrastive_loss_ratio:
                contrast_prob = self.softmax(event_logits.reshape(input_ids.size(0), -1, self.num_labels))
                if self.num_labels == 1:
                    contrastive_loss = self.cont_loss(contrast_prob.view(-1), event_labels.view(-1).float())
                else: 
                    contrastive_loss = self.cont_loss(contrast_prob.view(-1, self.num_labels), event_labels.view(-1))
                loss += self.contrastive_loss_ratio * contrastive_loss 
            if self.rank_loss_ratio:
                loss += self.rank_loss_ratio * event_rank_loss
            if self.num_labels == 1:
                logits = torch.stack((1-torch.sigmoid(logits), torch.sigmoid(logits)), dim = -1).squeeze(1)
            #print("7", datetime.now().time())
            return logits, loss
        if self.num_labels == 1:
            logits = torch.stack((1-torch.sigmoid(logits), torch.sigmoid(logits)), dim = -1).squeeze(1)
        return logits
    
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

# numnet for negative sampling
class NumNetMultitaskClassifierRoberta2(NumNetMultitaskClassifierRoberta):
    config_class = RobertaConfig
    # def __init__(self, config, mlp_hid=16, gcn_steps=4, max_question_length=50, use_gcn=False, event_loss_ratio=0.0, use_parser_tag=False,
    # contrastive_loss_ratio=0, dropout_prob = -1, rank_loss_ratio=0, num_labelss=2):
    #     super(NumNetMultitaskClassifierRoberta2, self).__init__(config, mlp_hid, gcn_steps, max_question_length, 
    #     use_gcn, event_loss_ratio, use_parser_tag, contrastive_loss_ratio, dropout_prob, rank_loss_ratio, num_labelss)

    def forward(self, input_ids, offsets, lengths, bpe_to_node, edges, question_len, attention_mask=None,
                token_type_ids=None, position_ids=None, head_mask=None, labels=None, events=None, 
                q_event_word_idx = None, q_tr_word_idx = None, parser_labels = None, same_p_mask=None):
        #print("1", datetime.now().time())
        outputs = self.roberta(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               head_mask=head_mask, output_hidden_states=True)

        sequence_output = self.dropout(outputs[0]) # last hidden states

        if same_p_mask:
            orig_instance_ids = [qid[0] for qid in same_p_mask]
            lengths_sum = [range(sum(lengths[:i]), sum(lengths[:i+1])) for i in range(len(lengths))]
            orig_instance_idxs = [lengths_sum[inst_id] for inst_id in orig_instance_ids]
            orig_labels = labels[list(chain(*orig_instance_idxs))]
            orig_events = events[list(chain(*orig_instance_idxs))]
            if self.rank_loss_ratio:
                # get original length, original input_id, original_seq_output, original_events    
                # calculate
                # orig_offsets = 
                # orig_lengths = 
                # orig_input_ids = 
                # orig_sequence_output = 
                x = self.lm_head.dense(sequence_output)
                x = self.gelu(x)
                x = self.lm_head.layer_norm(x)
                event_rank_loss = self.event_projector(input_ids, x, sequence_output, orig_events, orig_labels, offsets, lengths)
        else:
            orig_events = events
            orig_labels = labels        

        if self.use_gcn:
            sequence_output_list = [ item for item in outputs[2][-2:] ] # last 2 hidden states
            sequence_alg = self._gcn_input_proj(torch.cat([sequence_output_list[0], sequence_output_list[1]], dim=2))
            
            total_word_length = [len(h) for h in bpe_to_node]
            doc_len = [total_word_length[i]-question_len[i] for i in range(len(total_word_length))]
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
       
            #sequence_output = self.dropout(self._gcn_enc(self._proj_ln(sequence_output)))
            sequence_output = self._proj_ln(sequence_output) #TODO: i fixed this. should experiment again

        idx = 0
        vectors = []
        for b, l in enumerate(lengths):
            for i in range(l):
                vectors.append(sequence_output[b, offsets[idx], :].unsqueeze(0))
                idx += 1
        assert idx == sum(lengths)
        vectors = torch.cat(vectors, dim=0)

        outputs = self.act(self.linear1(vectors))
        logits = self.linear2(outputs)

        if same_p_mask:
            orig_logits = logits[list(chain(*orig_instance_idxs))]
        else:
            orig_logits = logits

        if labels is not None:
            if self.num_labels == 2:
                loss = self.loss_fct(orig_logits.view(-1, self.num_labels), orig_labels.view(-1))
            elif self.num_labels == 1:
                loss = self.loss_fct(orig_logits.view(-1), orig_labels.view(-1).float())
            if self.contrastive_loss_ratio and same_p_mask:
                contrastive_loss = 0
                for qids in same_p_mask:
                    neg_idxs = [lengths_sum[inst_id] for inst_id in qids]
                    neg_logits = logits[list(chain(*neg_idxs))]
                    neg_labels = labels[list(chain(*neg_idxs))]
                    neg_lengths = [lengths[qid] for qid in qids]
                    if not self.wo_events: # select only event indexes
                        neg_events = events[list(chain(*neg_idxs))]
                        # set event label again. get the longest
                        event_tmp = neg_events[0:neg_lengths[0]] # get first
                        start_length=0
                        for l in neg_lengths:
                            event_tmp = torch.logical_or(event_tmp, neg_events[start_length:start_length+l])
                            start_length+=l
                        event_tmp = event_tmp.repeat(len(neg_lengths)) 
                        events_mask = event_tmp.unsqueeze(-1).expand(-1,self.num_labels)
                        neg_logits = torch.masked_select(neg_logits,events_mask.bool()).reshape(-1, self.num_labels)
                        neg_labels = torch.masked_select(neg_labels, event_tmp.bool())
                    else:
                        neg_idxs = [lengths_sum[inst_id] for inst_id in qids]
                        neg_logits = logits[list(chain(*neg_idxs))]
                        neg_labels = labels[list(chain(*neg_idxs))]

                    temp_prob = self.softmax(neg_logits.reshape(len(qids), -1, self.num_labels))
                    if self.num_labels == 1:
                        contrastive_loss += self.cont_loss(temp_prob.view(-1), neg_labels.view(-1))
                    else:
                        contrastive_loss += self.cont_loss(temp_prob.view(-1, self.num_labels), neg_labels.view(-1))
                contrastive_loss = contrastive_loss / len(same_p_mask)
                loss += self.contrastive_loss_ratio * contrastive_loss
            if self.rank_loss_ratio:
                loss += self.rank_loss_ratio * event_rank_loss
            if self.num_labels == 1:
                orig_logits = torch.stack((1-torch.sigmoid(orig_logits), torch.sigmoid(orig_logits)), dim = -1).squeeze(1)
            return orig_logits, loss
        if self.num_labels == 1:
            orig_logits = torch.stack((1-torch.sigmoid(orig_logits), torch.sigmoid(orig_logits)), dim = -1).squeeze(1)
        return orig_logits


# for cl in training & eval
class NumNetMultitaskClassifierRoberta3(NumNetMultitaskClassifierRoberta):
    config_class = RobertaConfig
    def __init__(self, config, mlp_hid=16, gcn_steps=4, max_question_length=50, use_gcn=False, event_loss_ratio=0.0, use_parser_tag=False,
    contrastive_loss_ratio=0, dropout_prob = -1, rank_loss_ratio=0, num_labelss=2, wo_events=False):
        super(NumNetMultitaskClassifierRoberta3, self).__init__(config, mlp_hid, gcn_steps, max_question_length, 
        use_gcn, event_loss_ratio, use_parser_tag, contrastive_loss_ratio, dropout_prob, rank_loss_ratio, num_labelss, wo_events)
        if self.contrastive_loss_ratio:
           self._cont_shuffler = nn.GRU(input_size=config.hidden_size, hidden_size=config.hidden_size // 2, num_layers=2,
                                batch_first=True, dropout=dropout_prob, bidirectional=True)


    def forward(self, input_ids, offsets, lengths, bpe_to_node, edges, question_len, attention_mask=None,
                token_type_ids=None, position_ids=None, head_mask=None, labels=None, events=None, 
                q_event_word_idx = None, q_tr_word_idx = None, parser_labels = None, same_p_mask=None):
        outputs = self.roberta(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               head_mask=head_mask, output_hidden_states=True)

        sequence_output = self.dropout(outputs[0]) # last hidden states

        if same_p_mask:
            orig_instance_ids = [qid[0] for qid in same_p_mask]
            lengths_sum = [range(sum(lengths[:i]), sum(lengths[:i+1])) for i in range(len(lengths))]
            orig_instance_idxs = [lengths_sum[inst_id] for inst_id in orig_instance_ids]
            orig_labels = labels[list(chain(*orig_instance_idxs))]
            orig_lengths = [lengths[qid] for qid in orig_instance_ids]
        else:
            orig_labels = labels
            orig_lengths = lengths

        if self.use_gcn:
            sequence_output_list = [ item for item in outputs[2][-2:] ] # last 2 hidden states
            sequence_alg = self._gcn_input_proj(torch.cat([sequence_output_list[0], sequence_output_list[1]], dim=2))
            
            total_word_length = [len(h) for h in bpe_to_node]
            doc_len = [total_word_length[i]-question_len[i] for i in range(len(total_word_length))]
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
       
            sequence_output = self.dropout(self._gcn_enc(self._proj_ln(sequence_output)))

        idx = 0
        vectors = []
        for b, l in enumerate(lengths):
            for i in range(l):
                vectors.append(sequence_output[b, offsets[idx], :].unsqueeze(0))
                idx += 1
        assert idx == sum(lengths)

        vectors = torch.cat(vectors, dim=0)

        outputs = self.act(self.linear1(vectors))
        logits = self.linear2(outputs)

        if same_p_mask:
            orig_logits = logits[list(chain(*orig_instance_idxs))]
        else:
            orig_logits = logits

        if labels is not None:
            if self.num_labels == 2:
                loss = self.loss_fct(orig_logits.view(-1, self.num_labels), orig_labels.view(-1))
            return orig_logits, loss, orig_lengths, orig_labels
        return orig_logits, orig_lengths

# crossencoder
class NumNetMultitaskClassifierRoberta4(NumNetMultitaskClassifierRoberta):
    config_class = RobertaConfig
    def __init__(self, config, mlp_hid=16, gcn_steps=4, max_question_length=50, use_gcn=False, event_loss_ratio=0.0, use_parser_tag=False,
    contrastive_loss_ratio=0, dropout_prob = -1, rank_loss_ratio=0, num_labelss=2, wo_events=False, residual_connection=False, question_concat = False,
    deliberate = 0, ablation = 0, share=False, deliberate_ffn=2048):
        super(NumNetMultitaskClassifierRoberta4, self).__init__(config, mlp_hid, gcn_steps, max_question_length, 
        use_gcn, event_loss_ratio, use_parser_tag, contrastive_loss_ratio, dropout_prob, rank_loss_ratio, num_labelss, wo_events, question_concat)
        self.residual_connection = residual_connection
        if self.use_gcn:
            node_dim = config.hidden_size
            self._gcn_input_proj = nn.Linear(node_dim * 4, node_dim)
        self.deliberate = deliberate
        if self.deliberate:
            encoder_layer = TransformerEncoderLayerWithCrossAttention(config, dim_feedforward=deliberate_ffn, ablation=ablation)
            norm = nn.LayerNorm(config.hidden_size)
            self._cont_shuffler = nn.TransformerEncoder(encoder_layer, self.deliberate, norm)
        if share:
            self.linear3 = self.linear1
            self.linear4 = self.linear2
        else:
            self.linear3 = nn.Linear(config.hidden_size, mlp_hid)
            self.linear4 = nn.Linear(mlp_hid, self.num_labels)

    def forward(self, input_ids, offsets, lengths, bpe_to_node, edges, question_len, attention_mask=None,
                token_type_ids=None, position_ids=None, head_mask=None, labels=None, events=None, 
                q_event_word_idx = None, q_tr_word_idx = None, parser_labels = None, same_p_mask=None):
        #print("1", datetime.now().time())
        outputs = self.roberta(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               head_mask=head_mask, output_hidden_states=True)

        sequence_output = self.dropout(outputs[0]) # last hidden states
        #print("2", datetime.now().time())
        if self.use_parser_tag:
            new_sequence_output = torch.zeros((sequence_output.size(0), sequence_output.size(1), sequence_output.size(2)+parser_labels.size(2))).to(sequence_output.device)
            new_sequence_output[:,:,:sequence_output.size(-1)] = sequence_output
            idx = 0
            #vectors = []
            for b, l in enumerate(lengths):
                for i in range(l):
                    #torch.cat(sequence_output[b, offsets[idx], :],parser_labels[b, i, :])
                    new_sequence_output[b, offsets[idx], sequence_output.size(-1):] = parser_labels[b, i, :]
                    idx += 1
            assert idx == sum(lengths)
            sequence_output = new_sequence_output
            sequence_output = self.gelu(self.linear_p(new_sequence_output))

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
                    #sequence_output[b, pos] = torch.add(sequence_output[b, pos] ,d_node[b, i])       
            #sequence_output = self.dropout(self._gcn_enc(self._proj_ln(sequence_output)))
            sequence_output = self._proj_ln(sequence_output)
        
        if self.question_concat:
            # select encoded question and mean
            idx = 0
            vectors = []
            for b, l in enumerate(lengths):
                encoded_question = sequence_output[b,1:bpe_to_node[b][question_len[b]]].mean(0)
                for i in range(l):
                    vectors.append(torch.cat([sequence_output[b, offsets[idx], :].unsqueeze(0), encoded_question.unsqueeze(0)], dim=-1))
                    idx += 1
            assert idx == sum(lengths)
            vectors = torch.cat(vectors, dim=0)
            vectors = self.concatlayer(vectors)
        else:
            idx = 0
            vectors = []
            for b, l in enumerate(lengths):
                for i in range(l):
                    vectors.append(sequence_output[b, offsets[idx], :].unsqueeze(0))
                    idx += 1
            assert idx == sum(lengths)
            vectors = torch.cat(vectors, dim=0)

        if self.deliberate:
            vectors2 = vectors.reshape(input_ids.size(0), -1, vectors.size(-1))
            vectors2 = self._cont_shuffler(vectors2)

        outputs1 = self.act(self.linear1(vectors))
        logits1 = self.linear2(outputs1)
        
        if self.residual_connection:
            vectors2 = vectors2.reshape(-1, vectors2.size(-1))
            outputs = self.act(self.linear3(vectors + vectors2))
            logits = self.linear4(outputs).view(-1, self.num_labels)
        else:
            outputs = self.act(self.linear3(vectors2))
            logits = self.linear4(outputs).view(-1, self.num_labels)

        if self.contrastive_loss_ratio and not self.wo_events:
            # set event label again. get the longest
            event_tmp = events[0:lengths[0]] # get first
            start_length=0
            for l in lengths:
                event_tmp = torch.logical_or(event_tmp, events[start_length:start_length+l])
                start_length+=l
            event_lens = [sum(event_tmp)] * len(lengths)
            event_tmp = event_tmp.repeat(len(lengths)) 
            events_mask = event_tmp.unsqueeze(-1).expand(-1,self.num_labels)
            event_logits = torch.masked_select(logits,events_mask.bool()).reshape(-1, self.num_labels)
            event_labels = torch.masked_select(labels, event_tmp.bool())    
        else:
            event_logits = logits
            event_labels = labels      

        if labels is not None:
            if self.num_labels == 2:
                loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1)) + self.loss_fct(logits1.view(-1, self.num_labels), labels.view(-1))
                loss = loss / 2
            elif self.num_labels == 1:
                loss = self.loss_fct(logits.view(-1), labels.view(-1).float()) + self.loss_fct(logits1.view(-1), labels.view(-1).float())
            if self.event_loss_ratio:
            #     # event label is only given during training
                event_pred = self.event_detector(vectors)
                event_fct = CrossEntropyLoss()
                event_loss = event_fct(event_pred.view(-1,self.num_labels), events.view(-1))
                loss += self.event_loss_ratio * event_loss
            if self.contrastive_loss_ratio:
                contrast_prob = self.softmax(event_logits.reshape(input_ids.size(0), -1, self.num_labels))
                contrastive_loss = self.cont_loss(contrast_prob.view(-1, self.num_labels), event_labels.view(-1))
                loss += self.contrastive_loss_ratio * contrastive_loss
            return logits, loss
        return logits


class NumNetMultitaskClassifierBERT4(BertPreTrainedModel):
    def __init__(self, config, mlp_hid=16, gcn_steps=4, max_question_length=50, use_gcn=False, event_loss_ratio=0.0, use_parser_tag=False,
    contrastive_loss_ratio=0, dropout_prob = -1, rank_loss_ratio=0, num_labelss=2, wo_events=False, residual_connection=False, question_concat = False,
    deliberate = 0, ablation = 0, share=False, deliberate_ffn=2048):
        super(NumNetMultitaskClassifierBERT4, self).__init__(config)
        self.bert = BertModel(config)
        self.num_labels = num_labelss
        self.contrastive_loss_ratio = contrastive_loss_ratio
        self.wo_events = wo_events
        self.use_gcn = use_gcn
        if self.num_labels == 2:
            self.loss_fct = CrossEntropyLoss()
            self.softmax = nn.LogSoftmax(dim=0)
            if self.contrastive_loss_ratio:                
                self.cont_loss = NLLLoss()
        else:
            raise KeyError("num labels should be 2")

        dropout_prob = dropout_prob if dropout_prob >=0 else config.hidden_dropout_prob
        self.dropout = nn.Dropout(dropout_prob)
        self.linear1 = nn.Linear(config.hidden_size, mlp_hid)
        self.linear2 = nn.Linear(mlp_hid, self.num_labels)
        self.act = nn.Tanh()

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
        if share:
            self.linear3 = self.linear1
            self.linear4 = self.linear2
        else:
            self.linear3 = nn.Linear(config.hidden_size, mlp_hid)
            self.linear4 = nn.Linear(mlp_hid, self.num_labels)
        
        self.init_weights()

    def forward(self, input_ids, offsets, lengths, bpe_to_node, edges, question_len, attention_mask=None,
                token_type_ids=None, position_ids=None, head_mask=None, labels=None, events=None, 
                q_event_word_idx = None, q_tr_word_idx = None, parser_labels = None, same_p_mask=None):
        #print("1", datetime.now().time())
        outputs = self.bert(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               head_mask=head_mask, output_hidden_states=True)

        sequence_output = self.dropout(outputs[0]) # last hidden states

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
                    #sequence_output[b, pos] = torch.add(sequence_output[b, pos] ,d_node[b, i])       
            #sequence_output = self.dropout(self._gcn_enc(self._proj_ln(sequence_output)))
            sequence_output = self._proj_ln(sequence_output)
        
        idx = 0
        vectors = []
        for b, l in enumerate(lengths):
            for i in range(l):
                vectors.append(sequence_output[b, offsets[idx], :].unsqueeze(0))
                idx += 1
        assert idx == sum(lengths)
        vectors = torch.cat(vectors, dim=0)

        if self.deliberate:
            vectors2 = vectors.reshape(input_ids.size(0), -1, vectors.size(-1))
            vectors2 = self._cont_shuffler(vectors2)

        outputs1 = self.act(self.linear1(vectors))
        logits1 = self.linear2(outputs1)
        
        if self.deliberate:
            if self.residual_connection:
                vectors2 = vectors2.reshape(-1, vectors2.size(-1))
                outputs = self.act(self.linear3(vectors + vectors2))
                logits = self.linear4(outputs).view(-1, self.num_labels)
            else:
                outputs = self.act(self.linear3(vectors2))
                logits = self.linear4(outputs).view(-1, self.num_labels)
        else:
            logits = logits1

        if self.contrastive_loss_ratio and not self.wo_events:
            # set event label again. get the longest
            event_tmp = events[0:lengths[0]] # get first
            start_length=0
            for l in lengths:
                event_tmp = torch.logical_or(event_tmp, events[start_length:start_length+l])
                start_length+=l
            event_lens = [sum(event_tmp)] * len(lengths)
            event_tmp = event_tmp.repeat(len(lengths)) 
            events_mask = event_tmp.unsqueeze(-1).expand(-1,self.num_labels)
            event_logits = torch.masked_select(logits,events_mask.bool()).reshape(-1, self.num_labels)
            event_labels = torch.masked_select(labels, event_tmp.bool())    
        else:
            event_logits = logits
            event_labels = labels      

        if labels is not None:
            if self.num_labels == 2:
                loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1)) + self.loss_fct(logits1.view(-1, self.num_labels), labels.view(-1))
                loss = loss / 2
            elif self.num_labels == 1:
                loss = self.loss_fct(logits.view(-1), labels.view(-1).float()) + self.loss_fct(logits1.view(-1), labels.view(-1).float())
            # if self.event_loss_ratio:
            # #     # event label is only given during training
            #     event_pred = self.event_detector(vectors)
            #     event_fct = CrossEntropyLoss()
            #     event_loss = event_fct(event_pred.view(-1,self.num_labels), events.view(-1))
            #     loss += self.event_loss_ratio * event_loss
            if self.contrastive_loss_ratio:
                contrast_prob = self.softmax(event_logits.reshape(input_ids.size(0), -1, self.num_labels))
                contrastive_loss = self.cont_loss(contrast_prob.view(-1, self.num_labels), event_labels.view(-1))
                loss += self.contrastive_loss_ratio * contrastive_loss
            return logits, loss
        return logits
    
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



class NumNetMultitaskClassifierDeBERTa4(DebertaPreTrainedModel):
    #config_class = DebertaConfig
    def __init__(self, config, mlp_hid=16, gcn_steps=4, max_question_length=50, use_gcn=False, event_loss_ratio=0.0, use_parser_tag=False,
    contrastive_loss_ratio=0, dropout_prob = -1, rank_loss_ratio=0, num_labelss=2, wo_events=False, residual_connection=False, question_concat = False,
    deliberate = 0, ablation = 0, share=False, deliberate_ffn=2048):
        super(NumNetMultitaskClassifierDeBERTa4, self).__init__(config)
        self.deberta = DebertaModel(config)
        self.num_labels = num_labelss
        self.contrastive_loss_ratio = contrastive_loss_ratio
        self.wo_events = wo_events
        self.use_gcn = use_gcn
        if self.num_labels == 2:
            self.loss_fct = CrossEntropyLoss()
            self.softmax = nn.LogSoftmax(dim=0)
            if self.contrastive_loss_ratio:                
                self.cont_loss = NLLLoss()
        else:
            raise KeyError("num labels should be 2")

        dropout_prob = dropout_prob if dropout_prob >=0 else config.hidden_dropout_prob
        self.dropout = nn.Dropout(dropout_prob)
        self.linear1 = nn.Linear(config.hidden_size, mlp_hid)
        self.linear2 = nn.Linear(mlp_hid, self.num_labels)
        self.act = nn.Tanh()

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
        if share:
            self.linear3 = self.linear1
            self.linear4 = self.linear2
        else:
            self.linear3 = nn.Linear(config.hidden_size, mlp_hid)
            self.linear4 = nn.Linear(mlp_hid, self.num_labels)
        
        self.init_weights()

    def forward(self, input_ids, offsets, lengths, bpe_to_node, edges, question_len, attention_mask=None,
                token_type_ids=None, position_ids=None, head_mask=None, labels=None, events=None, 
                q_event_word_idx = None, q_tr_word_idx = None, parser_labels = None, same_p_mask=None):
        #print("1", datetime.now().time())
        outputs = self.deberta(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids, output_hidden_states=True)

        sequence_output = self.dropout(outputs[0]) # last hidden states

        if self.use_gcn:
            sequence_output_list = [item for item in outputs[1][-4:] ] # last 4 hidden states
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
                    #sequence_output[b, pos] = torch.add(sequence_output[b, pos] ,d_node[b, i])       
            #sequence_output = self.dropout(self._gcn_enc(self._proj_ln(sequence_output)))
            sequence_output = self._proj_ln(sequence_output)
        
        idx = 0
        vectors = []
        for b, l in enumerate(lengths):
            for i in range(l):
                vectors.append(sequence_output[b, offsets[idx], :].unsqueeze(0))
                idx += 1
        assert idx == sum(lengths)
        vectors = torch.cat(vectors, dim=0)

        if self.deliberate:
            vectors2 = vectors.reshape(input_ids.size(0), -1, vectors.size(-1))
            vectors2 = self._cont_shuffler(vectors2)

        outputs1 = self.act(self.linear1(vectors))
        logits1 = self.linear2(outputs1)
        if self.deliberate:
            if self.residual_connection:
                vectors2 = vectors2.reshape(-1, vectors2.size(-1))
                outputs = self.act(self.linear3(vectors + vectors2))
                logits = self.linear4(outputs).view(-1, self.num_labels)
            else:
                outputs = self.act(self.linear3(vectors2))
                logits = self.linear4(outputs).view(-1, self.num_labels)

        # if self.contrastive_loss_ratio and not self.wo_events:
        #     # set event label again. get the longest
        #     event_tmp = events[0:lengths[0]] # get first
        #     start_length=0
        #     for l in lengths:
        #         event_tmp = torch.logical_or(event_tmp, events[start_length:start_length+l])
        #         start_length+=l
        #     event_lens = [sum(event_tmp)] * len(lengths)
        #     event_tmp = event_tmp.repeat(len(lengths)) 
        #     events_mask = event_tmp.unsqueeze(-1).expand(-1,self.num_labels)
        #     event_logits = torch.masked_select(logits,events_mask.bool()).reshape(-1, self.num_labels)
        #     event_labels = torch.masked_select(labels, event_tmp.bool())    
        # else:
        #     event_logits = logits
        #     event_labels = labels   
        if self.contrastive_loss_ratio:
            event_logits = logits
            event_labels = labels   
        if labels is not None:
            if self.num_labels == 2:
                if self.deliberate:
                    loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1)) + self.loss_fct(logits1.view(-1, self.num_labels), labels.view(-1))
                    loss = loss / 2
                else:
                    loss = self.loss_fct(logits1.view(-1, self.num_labels), labels.view(-1))
                    logits = logits1
            elif self.num_labels == 1:
                loss = self.loss_fct(logits.view(-1), labels.view(-1).float()) + self.loss_fct(logits1.view(-1), labels.view(-1).float())
            # if self.event_loss_ratio:
            # #     # event label is only given during training
            #     event_pred = self.event_detector(vectors)
            #     event_fct = CrossEntropyLoss()
            #     event_loss = event_fct(event_pred.view(-1,self.num_labels), events.view(-1))
            #     loss += self.event_loss_ratio * event_loss
            if self.contrastive_loss_ratio:
                contrast_prob = self.softmax(event_logits.reshape(input_ids.size(0), -1, self.num_labels))
                contrastive_loss = self.cont_loss(contrast_prob.view(-1, self.num_labels), event_labels.view(-1))
                loss += self.contrastive_loss_ratio * contrastive_loss
            return logits, loss
        return logits
    
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


class NumNetMultitaskClassifierDeBERTaV24(DebertaV2PreTrainedModel):
    #config_class = DebertaConfig
    def __init__(self, config, mlp_hid=16, gcn_steps=4, max_question_length=50, use_gcn=False, event_loss_ratio=0.0, use_parser_tag=False,
    contrastive_loss_ratio=0, dropout_prob = -1, rank_loss_ratio=0, num_labelss=2, wo_events=False, residual_connection=False, question_concat = False,
    deliberate = 0, ablation = 0, share=False, deliberate_ffn=2048):
        super(NumNetMultitaskClassifierDeBERTaV24, self).__init__(config)
        self.deberta = DebertaV2Model(config)
        self.num_labels = num_labelss
        self.contrastive_loss_ratio = contrastive_loss_ratio
        self.wo_events = wo_events
        self.use_gcn = use_gcn
        if self.num_labels == 2:
            self.loss_fct = CrossEntropyLoss()
            self.softmax = nn.LogSoftmax(dim=0)
            if self.contrastive_loss_ratio:                
                self.cont_loss = NLLLoss()
        else:
            raise KeyError("num labels should be 2")

        dropout_prob = dropout_prob if dropout_prob >=0 else config.hidden_dropout_prob
        self.dropout = nn.Dropout(dropout_prob)
        self.linear1 = nn.Linear(config.hidden_size, mlp_hid)
        self.linear2 = nn.Linear(mlp_hid, self.num_labels)
        self.act = nn.Tanh()

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
        if share:
            self.linear3 = self.linear1
            self.linear4 = self.linear2
        else:
            self.linear3 = nn.Linear(config.hidden_size, mlp_hid)
            self.linear4 = nn.Linear(mlp_hid, self.num_labels)
        
        self.init_weights()

    def forward(self, input_ids, offsets, lengths, bpe_to_node, edges, question_len, attention_mask=None,
                token_type_ids=None, position_ids=None, head_mask=None, labels=None, events=None, 
                q_event_word_idx = None, q_tr_word_idx = None, parser_labels = None, same_p_mask=None):
        #print("1", datetime.now().time())
        outputs = self.deberta(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids, output_hidden_states=True)

        sequence_output = self.dropout(outputs[0]) # last hidden states

        if self.use_gcn:
            sequence_output_list = [item for item in outputs[1][-4:] ] # last 4 hidden states
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
        
        idx = 0
        vectors = []
        for b, l in enumerate(lengths):
            for i in range(l):
                vectors.append(sequence_output[b, offsets[idx], :].unsqueeze(0))
                idx += 1
        assert idx == sum(lengths)
        vectors = torch.cat(vectors, dim=0)

        if self.deliberate:
            vectors2 = vectors.reshape(input_ids.size(0), -1, vectors.size(-1))
            vectors2 = self._cont_shuffler(vectors2)

        outputs1 = self.act(self.linear1(vectors))
        logits1 = self.linear2(outputs1)
        if self.deliberate:
            if self.residual_connection:
                vectors2 = vectors2.reshape(-1, vectors2.size(-1))
                outputs = self.act(self.linear3(vectors + vectors2))
                logits = self.linear4(outputs).view(-1, self.num_labels)
            else:
                outputs = self.act(self.linear3(vectors2))
                logits = self.linear4(outputs).view(-1, self.num_labels)
        else:
            logits = logits1

        event_logits = logits
        event_labels = labels     

        if labels is not None:
            if self.num_labels == 2:
                if self.deliberate:
                    loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1)) + self.loss_fct(logits1.view(-1, self.num_labels), labels.view(-1))
                    loss = loss / 2
                else:
                    loss = self.loss_fct(logits1.view(-1, self.num_labels), labels.view(-1))
                    logits = logits1
            elif self.num_labels == 1:
                loss = self.loss_fct(logits.view(-1), labels.view(-1).float()) + self.loss_fct(logits1.view(-1), labels.view(-1).float())
            if self.contrastive_loss_ratio:
                contrast_prob = self.softmax(event_logits.reshape(input_ids.size(0), -1, self.num_labels))
                contrastive_loss = self.cont_loss(contrast_prob.view(-1, self.num_labels), event_labels.view(-1))
                loss += self.contrastive_loss_ratio * contrastive_loss
            return logits, loss
        return logits
    
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

        qq_graph = graph[:, :max_q_word_length, :max_q_word_length]
        dd_graph = graph[:, self.max_question_length:, self.max_question_length:]
        qd_graph = graph[:, :max_q_word_length, self.max_question_length:]
        dq_graph=qd_graph.transpose(1,2)
        
        return qq_graph, dq_graph, dd_graph, qd_graph

class FFNLayer(nn.Module):
    def __init__(self, input_dim, intermediate_dim, output_dim, dropout, layer_norm=True):
        super(FFNLayer, self).__init__()
        self.fc1 = nn.Linear(input_dim, intermediate_dim)
        if layer_norm:
            self.ln = nn.LayerNorm(intermediate_dim)
        else:
            self.ln = None
        self.dropout_func = nn.Dropout(dropout)
        self.fc2 = nn.Linear(intermediate_dim, output_dim)

    def forward(self, input):
        inter = self.fc1(self.dropout_func(input))
        inter_act = nn.GELU(inter)
        if self.ln:
            inter_act = self.ln(inter_act)
        return self.fc2(inter_act)

class ResidualGRU(nn.Module):
    def __init__(self, hidden_size, dropout=0.1, num_layers=2):
        super(ResidualGRU, self).__init__()
        self.enc_layer = nn.GRU(input_size=hidden_size, hidden_size=hidden_size // 2, num_layers=num_layers,
                                batch_first=True, dropout=dropout, bidirectional=True)
        self.enc_ln = nn.LayerNorm(hidden_size)

    def forward(self, input):
        output, _ = self.enc_layer(input)
        return self.enc_ln(output + input)

class EventProjector(nn.Module):
    def __init__(self, hidden_size):
        super(EventProjector, self).__init__()
        self.event_projector = nn.Linear(hidden_size, hidden_size)
        self.mask_token_id = 50264  # 'mask token id'
        
    def forward(self, input_ids, q_event_output, sequence_output, events, labels, offsets, lengths):
        event_clr_loss = []
        #TODO: select mask indexs
        mask_mask = (input_ids == self.mask_token_id).unsqueeze(-1).expand(-1,-1,sequence_output.size(-1)) # mask token id of roberta    
        sequence_output = self.event_projector(sequence_output)
        q_event_output = self.event_projector(q_event_output)
        for bidx in range(input_ids.size(0)):
            events_i = events[sum(lengths[:bidx]):sum(lengths[:bidx+1])]
            labels_i = labels[sum(lengths[:bidx]):sum(lengths[:bidx+1])]
            if sum(labels_i) == 0: continue
            is_masked = mask_mask[bidx].nonzero(as_tuple=True)
            if is_masked[0].size(0) == 0: continue # questions like how, when, why
            # x = from question
            # FIXME: seed 7, 241, error
            x = q_event_output[bidx].masked_select(mask_mask[bidx]).reshape(1, sequence_output.size(-1))
            #x = self.event_projector(x)
            # y = from passage
            idx = 0
            ys = []
            for i in range(lengths[bidx]):
                ys.append(sequence_output[bidx, offsets[idx], :].unsqueeze(0))
                idx += 1
            ys = torch.cat(ys, dim=0)
            #ys = self.event_projector(ys)
            numerator = torch.masked_select(ys, labels_i.unsqueeze(-1).expand(-1,ys.size(-1)).bool()).reshape(-1, ys.size(-1))
            denominator = torch.masked_select(ys, events_i.unsqueeze(-1).expand(-1,ys.size(-1)).bool()).reshape(-1, ys.size(-1))
            
            #event_rep = F.normalize(event_rep, dim=-1)  # (36, 1024)
            temperature = 1.0
            n_similarity = torch.exp(F.cosine_similarity(x, numerator, dim=1) / temperature)  # (batch, 36, 36)
            d_similarity = torch.exp(F.cosine_similarity(x, denominator, dim=1) / temperature)  # (batch, 36, 36)
            clr_loss = -torch.log(torch.sum(n_similarity)/torch.sum(d_similarity))
            event_clr_loss.append(clr_loss)
        if len(event_clr_loss) > 0:
            event_clr_loss = sum(event_clr_loss) / len(event_clr_loss)
        else:
            event_clr_loss = torch.tensor(0.0)
        
        return event_clr_loss


class NumNetMultitaskClassifierElectra4(ElectraPreTrainedModel):
    #config_class = DebertaConfig
    def __init__(self, config, mlp_hid=16, gcn_steps=4, max_question_length=50, use_gcn=False, event_loss_ratio=0.0, use_parser_tag=False,
    contrastive_loss_ratio=0, dropout_prob = -1, rank_loss_ratio=0, num_labelss=2, wo_events=False, residual_connection=False, question_concat = False,
    deliberate = 0, ablation = 0, share=False, deliberate_ffn=2048):
        super(NumNetMultitaskClassifierElectra4, self).__init__(config)
        self.electra = ElectraModel(config)
        self.num_labels = num_labelss
        self.contrastive_loss_ratio = contrastive_loss_ratio
        self.wo_events = wo_events
        self.use_gcn = use_gcn
        if self.num_labels == 2:
            self.loss_fct = CrossEntropyLoss()
            self.softmax = nn.LogSoftmax(dim=0)
            if self.contrastive_loss_ratio:                
                self.cont_loss = NLLLoss()
        else:
            raise KeyError("num labels should be 2")

        dropout_prob = dropout_prob if dropout_prob >=0 else config.hidden_dropout_prob
        self.dropout = nn.Dropout(dropout_prob)
        self.linear1 = nn.Linear(config.hidden_size, mlp_hid)
        self.linear2 = nn.Linear(mlp_hid, self.num_labels)
        self.act = nn.Tanh()

        self.residual_connection = residual_connection
        if self.use_gcn:
            node_dim = config.hidden_size
            self._gcn_input_proj = nn.Linear(node_dim * 4, node_dim)
            self._gcn = GCN(node_dim=node_dim, iteration_steps=gcn_steps)
            self._proj_ln = nn.LayerNorm(node_dim)
            self._gcn_enc = ResidualGRU(config.hidden_size, config.attention_probs_dropout_prob, 2)
            self.max_question_length = max_question_length

        self.deliberate = deliberate
        if self.deliberate:
            encoder_layer = TransformerEncoderLayerWithCrossAttention(config, dim_feedforward=deliberate_ffn, ablation=ablation)
            norm = nn.LayerNorm(config.hidden_size)
            self._cont_shuffler = nn.TransformerEncoder(encoder_layer, self.deliberate, norm)
        if share:
            self.linear3 = self.linear1
            self.linear4 = self.linear2
        else:
            self.linear3 = nn.Linear(config.hidden_size, mlp_hid)
            self.linear4 = nn.Linear(mlp_hid, self.num_labels)
        
        self.init_weights()

    def forward(self, input_ids, offsets, lengths, bpe_to_node, edges, question_len, attention_mask=None,
                token_type_ids=None, position_ids=None, head_mask=None, labels=None, events=None, 
                q_event_word_idx = None, q_tr_word_idx = None, parser_labels = None, same_p_mask=None):
        #print("1", datetime.now().time())
        outputs = self.electra(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids, output_hidden_states=True)

        sequence_output = self.dropout(outputs[0]) # last hidden states

        if self.use_gcn:
            sequence_output_list = [item for item in outputs[1][-4:] ] # last 4 hidden states
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
        
        idx = 0
        vectors = []
        for b, l in enumerate(lengths):
            for i in range(l):
                vectors.append(sequence_output[b, offsets[idx], :].unsqueeze(0))
                idx += 1
        assert idx == sum(lengths)
        vectors = torch.cat(vectors, dim=0)

        if self.deliberate:
            vectors2 = vectors.reshape(input_ids.size(0), -1, vectors.size(-1))
            vectors2 = self._cont_shuffler(vectors2)

        outputs1 = self.act(self.linear1(vectors))
        logits1 = self.linear2(outputs1)
        if self.deliberate:
            if self.residual_connection:
                vectors2 = vectors2.reshape(-1, vectors2.size(-1))
                outputs = self.act(self.linear3(vectors + vectors2))
                logits = self.linear4(outputs).view(-1, self.num_labels)
            else:
                outputs = self.act(self.linear3(vectors2))
                logits = self.linear4(outputs).view(-1, self.num_labels)    

        if labels is not None:
            if self.num_labels == 2:
                if self.deliberate:
                    loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1)) + self.loss_fct(logits1.view(-1, self.num_labels), labels.view(-1))
                    loss = loss / 2
                else:
                    loss = self.loss_fct(logits1.view(-1, self.num_labels), labels.view(-1))
                    logits = logits1
            elif self.num_labels == 1:
                loss = self.loss_fct(logits.view(-1), labels.view(-1).float()) + self.loss_fct(logits1.view(-1), labels.view(-1).float())
            return logits, loss
        return logits
    
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