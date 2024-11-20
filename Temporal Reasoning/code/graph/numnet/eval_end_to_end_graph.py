
# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import argparse
import random
from tqdm import tqdm, trange
import pickle
import numpy as np
import json
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from collections import Counter
from transformers import *
from models import MultitaskClassifier, MultitaskClassifierRoberta
from models_graph import (NumNetMultitaskClassifierRoberta, NumNetMultitaskClassifierRoberta2, NumNetMultitaskClassifierRoberta4,
                            NumNetMultitaskClassifierDeBERTa4, NumNetMultitaskClassifierElectra4, NumNetMultitaskClassifierBERT4,
                            NumNetMultitaskClassifierDeBERTaV24)
from models_graph_prev import GraphMultitaskClassifierRoberta, GraphMultitaskClassifierRobertaV2
from optimization import *
from collections import defaultdict
from utils import *
from utils_gnn import *
import sys
from pathlib import Path
from collate_fn import GraphDataCollator, GraphDataset, OtrDataset, GraphDataCollatorCont

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

PYTORCH_PRETRAINED_ROBERTA_CACHE = Path(os.getenv('PYTORCH_PRETRAINED_ROBERTA_CACHE',
                                               Path.home() / '.pytorch_pretrained_roberta'))

def main():
    parser = argparse.ArgumentParser()
    ## Required parameters                                                                                         
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .json files (or other data files) for the task.")
    parser.add_argument("--model", default=None, type=str, required=True,
                        help="pre-trained model selected in the list: roberta-base, "
                             "roberta-large, bert-base, bert-large. ")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--model_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The directory where the trained model are saved")
    parser.add_argument("--file_suffix",
                        default=None,
                        type=str,
                        required=True,
                        help="Suffix of filename")
    ## Other parameters                                                                                            
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--mlp_hid_size",
                        default=64,
                        type=int,
                        help="hid dimension for MLP layer.")
    parser.add_argument("--eval_ratio",
                        default=0.5,
                        type=float,
                        help="portion of data for evaluation")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=7,
                        help="random seed for initialization")
    parser.add_argument("--gcn_steps",
                        type=int,
                        default=4)
    parser.add_argument("--use_gcn",
                        action='store_true',
                        help="Whether to use gcn.")
    parser.add_argument("--use_qdgat",
                        default = 0,
                        type = int,
                        help="Whether to use qdgat.")
    parser.add_argument("--max_question_length",
                        default=35,
                        type=int,
                        help="The maximum total question word length after spacy tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--question_class_file",
                        default=None,
                        type=str,
                        help="The name of the question class file for analysis")
    parser.add_argument('--contrastive_loss_ratio',
                        type=float,
                        default=0,
                        help="0: baseline 1: in-contrast-question 2: in-same-passage")
    parser.add_argument('--dropout_prob',
                        type=float,
                        default=-1,
                        help="dropout prob")             
    parser.add_argument("--event_loss_ratio",
                        type=int,
                        default=0)
    parser.add_argument("--use_parser_tag",
                        action="store_true")
    parser.add_argument("--graph_construction",
                        default = 0,
                        type = int,
                        help="Whether to use qdgat.")
    parser.add_argument("--num_labels",
                        default = 2,
                        type = int,
                        help="num labels (1 or 2).")
    parser.add_argument('--use_event_chain',
                        nargs='+',
                        #action="store_true",
                        help = "use event chain output")
    parser.add_argument('--rank_loss_ratio',
                        type=float,
                        default=0,
                        help="event rank loss")
    parser.add_argument('--n_negs',
                        type=int,
                        default=0,
                        help="number of negatives when negative sampling")
    parser.add_argument("--wo_events",
                        action='store_true',
                        help="Without events")
    parser.add_argument("--deliberate", default=0, type=int, help="deliberate")
    parser.add_argument("--deliberate_ffn", default=2048, type=int, help="deliberate_ffn")
    parser.add_argument("--residual_connection",
                        action="store_true")
    parser.add_argument("--question_concat",
                        action="store_true")
    parser.add_argument("--debug",
                        action="store_true")
    parser.add_argument('--abl',
                        type=int,
                        default=0,
                        help="ablation level")

    args = parser.parse_args()
    print(args)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()

    logger.info("device: {} n_gpu: {}".format(device, n_gpu))
    

    random.seed(args.seed)
    np.random.seed(args.seed)
    if is_torch_available():
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)

    task_name = args.task_name.lower()

    logger.info("current task is " + str(task_name))

    label_map = {0: 'Negative', 1: 'Positive'}
    num_classes = len(label_map)
    model_state_dict = torch.load(args.model_dir + "pytorch_model.bin")

    if 'roberta' in args.model:
        tokenizer = RobertaTokenizer.from_pretrained(args.model)
        cache_dir = f"{PYTORCH_PRETRAINED_ROBERTA_CACHE}/distributed_1"
        if args.n_negs:
            logger.info(f"numnet2")
            model = NumNetMultitaskClassifierRoberta2.from_pretrained(args.model, state_dict = model_state_dict, mlp_hid=args.mlp_hid_size, gcn_steps=args.gcn_steps,
                                                            max_question_length=args.max_question_length, use_gcn=args.use_gcn, event_loss_ratio=args.event_loss_ratio, use_parser_tag=args.use_parser_tag,
                                                            contrastive_loss_ratio=args.contrastive_loss_ratio, dropout_prob=args.dropout_prob, rank_loss_ratio=args.rank_loss_ratio,
                                                            num_labelss=args.num_labels, wo_events = args.wo_events)
        elif args.deliberate:
            logger.info(f"numnet4")
            model = NumNetMultitaskClassifierRoberta4.from_pretrained(args.model, state_dict = model_state_dict, mlp_hid=args.mlp_hid_size, gcn_steps=args.gcn_steps,
                                                            max_question_length=args.max_question_length, use_gcn=args.use_gcn, event_loss_ratio=args.event_loss_ratio, use_parser_tag=args.use_parser_tag,
                                                            contrastive_loss_ratio=args.contrastive_loss_ratio, dropout_prob=args.dropout_prob, rank_loss_ratio=args.rank_loss_ratio,
                                                            num_labelss=args.num_labels, wo_events = args.wo_events, residual_connection=args.residual_connection, question_concat = args.question_concat, 
                                                            deliberate = args.deliberate, ablation=args.abl,
                                                            deliberate_ffn=args.deliberate_ffn)
        else:
            logger.info(f"numnet")
            model = NumNetMultitaskClassifierRoberta.from_pretrained(args.model, state_dict = model_state_dict, mlp_hid=args.mlp_hid_size, gcn_steps=args.gcn_steps,
                                                                    max_question_length=args.max_question_length, use_gcn=args.use_gcn, event_loss_ratio=args.event_loss_ratio, use_parser_tag=args.use_parser_tag,
                                                                    contrastive_loss_ratio=args.contrastive_loss_ratio, dropout_prob=args.dropout_prob, rank_loss_ratio=args.rank_loss_ratio,
                                                                    num_labelss=args.num_labels, wo_events = args.wo_events, question_concat = args.question_concat)        
    elif 'deberta' in args.model:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if "-v2" in args.model or "-v3" in args.model:
            logger.info(f"debertav2")
            model = NumNetMultitaskClassifierDeBERTaV24.from_pretrained(args.model, state_dict = model_state_dict, mlp_hid=args.mlp_hid_size, gcn_steps=args.gcn_steps,
                                                            max_question_length=args.max_question_length, use_gcn=args.use_gcn, event_loss_ratio=args.event_loss_ratio, use_parser_tag=args.use_parser_tag,
                                                            contrastive_loss_ratio=args.contrastive_loss_ratio, dropout_prob=args.dropout_prob, rank_loss_ratio=args.rank_loss_ratio,
                                                            num_labelss=args.num_labels, wo_events = args.wo_events, residual_connection=args.residual_connection, question_concat = args.question_concat, 
                                                            deliberate = args.deliberate, ablation=args.abl, deliberate_ffn=args.deliberate_ffn)
        else:   
            logger.info(f"deberta")
            model = NumNetMultitaskClassifierDeBERTa4.from_pretrained(args.model, state_dict = model_state_dict, mlp_hid=args.mlp_hid_size, gcn_steps=args.gcn_steps,
                                                            max_question_length=args.max_question_length, use_gcn=args.use_gcn, event_loss_ratio=args.event_loss_ratio, use_parser_tag=args.use_parser_tag,
                                                            contrastive_loss_ratio=args.contrastive_loss_ratio, dropout_prob=args.dropout_prob, rank_loss_ratio=args.rank_loss_ratio,
                                                            num_labelss=args.num_labels, wo_events = args.wo_events, residual_connection=args.residual_connection, question_concat = args.question_concat, 
                                                            deliberate = args.deliberate, ablation=args.abl, deliberate_ffn=args.deliberate_ffn)
    else:  
        tokenizer = BertTokenizer.from_pretrained(args.model)
        logger.info(f"bert")
        model = NumNetMultitaskClassifierBERT4.from_pretrained(args.model, state_dict = model_state_dict, mlp_hid=args.mlp_hid_size, gcn_steps=args.gcn_steps,
                                                            max_question_length=args.max_question_length, use_gcn=args.use_gcn, event_loss_ratio=args.event_loss_ratio, use_parser_tag=args.use_parser_tag,
                                                            contrastive_loss_ratio=args.contrastive_loss_ratio, dropout_prob=args.dropout_prob, rank_loss_ratio=args.rank_loss_ratio,
                                                            num_labelss=args.num_labels, wo_events = args.wo_events, residual_connection=args.residual_connection, question_concat = args.question_concat, 
                                                            deliberate = args.deliberate, ablation=args.abl, deliberate_ffn=args.deliberate_ffn)
    
    model.to(device)
    
    
    for eval_file in ['dev']:
        pred_out_dict = defaultdict(dict)
        print("=" * 50 + "Evaluating %s" % eval_file + "="* 50)
        eval_data = load_data(args.data_dir, "individual_%s" % eval_file, args.file_suffix)
        if 'roberta' in args.model:
            eval_features = convert_to_features_roberta_graph2(eval_data, tokenizer, instance=False,
                                                                max_length=args.max_seq_length, end_to_end=True,
                                                                max_question_length=args.max_question_length,
                                                                evaluation=True, is_qdgat=args.use_qdgat,
                                                                graph_construction=args.graph_construction,
                                                                event_chain_files=args.use_event_chain,
                                                                mask_what = bool(args.rank_loss_ratio))
        if 'deberta' in args.model:
            eval_features = convert_to_features_deberta_graph2(eval_data, tokenizer, instance=False,
                                                                max_length=args.max_seq_length, end_to_end=True,
                                                                max_question_length=args.max_question_length,
                                                                evaluation=True, is_qdgat=args.use_qdgat,
                                                                graph_construction=args.graph_construction,
                                                                event_chain_files=args.use_event_chain,
                                                                mask_what = bool(args.rank_loss_ratio))
        else:
            eval_features = convert_to_features_bert_graph2(eval_data, tokenizer, instance=False,
                                                                max_length=args.max_seq_length, end_to_end=True,
                                                                max_question_length=args.max_question_length,
                                                                evaluation=True, is_qdgat=args.use_qdgat,
                                                                graph_construction=args.graph_construction,
                                                                event_chain_files=args.use_event_chain,
                                                                mask_what = bool(args.rank_loss_ratio))        


        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_features))
        logger.info("  Batch size = %d", args.eval_batch_size)

        eval_offsets = select_field(eval_features, 'offset')
        eval_labels  = select_field(eval_features, 'label')
        # eval_key_indices = torch.tensor(list(range(len(eval_labels))), dtype=torch.long)
        eval_events = select_field(eval_features, 'events')
        eval_key_indices = list(range(len(eval_labels)))
        #eval_event_indices = list(range(len(eval_events)))

        # collect unique question ids for EM calculation
        question_ids_all = select_field(eval_features, 'question_id')
        question_ids = [q for i, q in enumerate(question_ids_all) for x in range(len(eval_labels[i]))]
        # collect unique question culster for EM-cluster calculation                          
        question_cluster = select_field(eval_features, 'question_cluster')
        question_cluster_size = select_field(eval_features, 'cluster_size')
        eval_idv_answers = select_field(eval_features, 'individual_answers')
        eval_passages = select_field(eval_features, 'passage')
        eval_questions = select_field(eval_features, 'question')

        if (args.contrastive_loss_ratio or args.deliberate) and not args.n_negs: #dynamic batch
            eval_dataset = OtrDataset(
                eval_features, args.eval_batch_size, evaluation=True)
            eval_graph_data_collator = GraphDataCollator(
                device=device, batch_size=args.eval_batch_size, evaluation=True, contrastive_loss=(args.contrastive_loss_ratio or args.deliberate))
            eval_dataloader = DataLoader(
                eval_dataset, shuffle=False, collate_fn=eval_graph_data_collator, batch_size=1)  # batch_size 1 -> dynamic batch size in dataloader
        else:
            eval_dataset = GraphDataset(
                eval_features, eval_key_indices)
            eval_graph_data_collator = GraphDataCollator(device=device, n_negs=args.n_negs, all_dataset=eval_dataset, evaluation=True)
            eval_dataloader = DataLoader(
                eval_dataset, shuffle=False, collate_fn=eval_graph_data_collator, batch_size=args.eval_batch_size)

        eval_loss, eval_accuracy, best_eval_f1, nb_eval_examples, nb_eval_steps = 0.0, 0.0, 0.0, 0, 0
        all_preds, all_golds, max_f1s, macro_f1s = [], [], [], []
        max_precisions, max_recalls = [], []
        #max_f1s_contrast, max_f1s_wo_contrast = [], []
        f1_dist = defaultdict(list)
        em_counter = 0
        em_cluster_agg, em_cluster_relaxed, f1_cluster_80 = {}, {}, {}
        # f1_cluster_70 = {}

        if args.question_class_file:
            print("load question class file")
            question_class = json.load(open(args.question_class_file))
            if type(list(question_class.values())[0]) == list:
                class_total = list(set([e for v in question_class.values() for e in v]))
                class_total.append("unclassified")
            else:
                class_total = list(set(question_class.values()))
            class_total = {k : 0 for k in class_total}
            em_class_counter = {k:0 for k in class_total.keys()}
            f1_class_counter = {k:list() for k in class_total.keys()}

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            with torch.no_grad():
                input_ids, input_masks, segment_ids, instance_indices, head_masks, \
                    question_len, edges, q_tr_word_idx, q_event_word_idx, same_p_mask = batch             
                offsets, labels, lengths = flatten_answers([(eval_labels[i], eval_offsets[i])
                                                            for i in instance_indices])
                _, events, _ = flatten_answers([(eval_events[i], eval_offsets[i])
                                                            for i in instance_indices])
                all_golds.extend(labels)
                labels = torch.tensor(labels).to(device)
                events = torch.tensor(events).to(device)
                
                logits, tmp_eval_loss = model(input_ids, offsets, lengths, edges=edges,
                                              bpe_to_node=head_masks, question_len=question_len, token_type_ids=segment_ids,
                                              attention_mask=input_masks, labels=labels, events = events, 
                                              q_tr_word_idx = q_tr_word_idx, q_event_word_idx = q_event_word_idx, same_p_mask = same_p_mask)
                    
                logits = logits.detach().cpu().numpy()
                labels = labels.to('cpu').numpy()

                nb_eval_examples += labels.shape[0]
                nb_eval_steps += 1

                batch_preds = np.argmax(logits, axis=1)
                bi = 0
                for l, idx in enumerate(instance_indices):
                    pred = [batch_preds[bi + li] for li in range(lengths[l])]
                    pred_names = [label_map[p] for p in pred]
                    gold_names = [label_map[labels[bi + li]] for li in range(lengths[l])]
                    is_em = (pred_names == gold_names)
                    if sum([labels[bi + li] for li in range(lengths[l])]) == 0 and sum(pred) == 0:
                        macro_f1s.append(1.0)
                    else:
                        macro_f1s.append(cal_f1(pred_names, gold_names, {v:k for k,v in label_map.items()}))
                        
                    max_f1, instance_matched = 0, 0
                    max_precision, max_recall = 0, 0
                    for gold in eval_idv_answers[idx]:
                        label_names = [label_map[l] for l in gold]
                        if pred_names == label_names: instance_matched = 1
                        if sum(gold) == 0 and sum(pred) == 0:
                            f1 = 1.0
                            precision, recall = 1.0, 1.0
                        else:
                            f1 = cal_f1(pred_names, label_names, {v:k for k,v in label_map.items()})
                            precision, recall = cal_pre_recall(pred_names, label_names, {v:k for k,v in label_map.items()})
                        if f1 >= max_f1:
                            max_f1 = f1
                            max_precision = precision
                            max_recall = recall
                            key = len(gold)

                    if question_cluster_size[idx] > 1:
                        if question_cluster[idx] not in em_cluster_agg:
                            em_cluster_agg[question_cluster[idx]] = 1
                        if is_em == 0: em_cluster_agg[question_cluster[idx]] = 0
                            
                        if question_cluster[idx] not in em_cluster_relaxed:
                            em_cluster_relaxed[question_cluster[idx]] = 1
                        if instance_matched == 0: em_cluster_relaxed[question_cluster[idx]] = 0

                        if question_cluster[idx] not in f1_cluster_80:
                            f1_cluster_80[question_cluster[idx]] = 1
                        if max_f1 < 0.8: 
                            f1_cluster_80[question_cluster[idx]] = 0

                    question = eval_questions[idx]
                    gold_passage = eval_passages[idx]
                    pred_events = [gold_passage[pi] for pi, p in enumerate(pred) if p > 0]
                    gold_events = [gold_passage[li] for li in range(lengths[l]) if labels[bi + li] > 0]
                    if pred_out_dict[str(gold_passage)].get(question_cluster[idx]) is None:
                        pred_out_dict[str(gold_passage)][question_cluster[idx]] = list()
                    pred_out_dict[str(gold_passage)][question_cluster[idx]].append({"question": question,
                                                                "pred": pred_events, 
                                                                "gold": gold_events,
                                                                "max_f1": max_f1,
                                                                "em": instance_matched
                                                                })      

                    bi += lengths[l]
                    max_f1s.append(max_f1)
                    max_precisions.append(max_precision)
                    max_recalls.append(max_recall)
                    em_counter += instance_matched
                    f1_dist[key].append(max_f1)

                    if args.question_class_file :
                        if type(question_class[question_ids_all[idx]]) == list:
                            if len(question_class[question_ids_all[idx]]) == 0:
                                cl = 'unclassified'
                                em_class_counter[cl] += instance_matched
                                class_total[cl] += 1
                                f1_class_counter[cl].append(max_f1)
                            else:
                                for cl in question_class[question_ids_all[idx]]:
                                    em_class_counter[cl] += instance_matched
                                    class_total[cl] += 1
                                    f1_class_counter[cl].append(max_f1)
                        else:
                            em_class_counter[question_class[question_ids_all[idx]]] += instance_matched
                            class_total[question_class[question_ids_all[idx]]] += 1
                            f1_class_counter[question_class[question_ids_all[idx]]].append(max_f1)


                all_preds.extend(batch_preds)

        assert len(em_cluster_relaxed) == len(em_cluster_agg)
        assert len(f1_cluster_80) == len(em_cluster_agg) 
        
        em_cluster_relaxed_res = sum(em_cluster_relaxed.values()) / len(em_cluster_relaxed)
        em_cluster_agg_res = sum(em_cluster_agg.values()) / len(em_cluster_agg)
        f1_cluster_80_res = sum(f1_cluster_80.values()) / len(f1_cluster_80)

        label_names = [label_map[l] for l in all_golds]
        pred_names = [label_map[p] for p in all_preds]
        
        # question_id is also flattened
        em = exact_match(question_ids, label_names, pred_names)
        eval_pos_f1 = cal_f1(pred_names, label_names, {v:k for k,v in label_map.items()})

        print("the current eval positive class Micro F1 (Agg) is: %.4f" % eval_pos_f1)
        print("the current eval positive class Macro F1 (Relaxed) is: %.4f" % np.mean(max_f1s))
        print("the current eval positive class precision (Relaxed) is: %.4f" % np.mean(max_precisions))
        print("the current eval positive class recall (Relaxed) is: %.4f" % np.mean(max_recalls))
        print("the current eval positive class Macro F1 (Agg) is: %.4f" % np.mean(macro_f1s))

        print("the current eval exact match (Agg) ratio is: %.4f" % em)
        print("the current eval exact match ratio (Relaxed) is: %.4f" % (em_counter / len(eval_features)))

        print("%d Clusters" % len(em_cluster_relaxed))
        print("the current eval clustered EM (Agg) is: %.4f" % (em_cluster_agg_res))
        print("the current eval clustered EM (Relaxed) is: %.4f" % (em_cluster_relaxed_res))
        print("the current eval clusrered F1 (max>=0.8) is: %.4f" % (f1_cluster_80_res))


        if args.question_class_file:
            print(class_total)
            print(f"total {sum(class_total.values())} questions")
            #print(em_class_counter)
            #print(f1_class_counter)
            
            print(" , ".join([str(k)+" "+str(total) for k, total in sorted(class_total.items()) if total > 0]))
            print(" , ".join([str(round(em_class_counter[k] / total, 4)) for k, total in sorted(class_total.items()) if total > 0]))
            print("----------------------------------------------------------------------------------------------------------------")
            print(" , ".join([str(round(np.mean(f1_class_counter[k]), 4)) for k, total in sorted(class_total.items()) if total > 0]))

        json.dump(pred_out_dict, open(f"{args.model_dir}/generated_dev.json", 'w'), indent=2)
            
if __name__ == "__main__":
    main()
