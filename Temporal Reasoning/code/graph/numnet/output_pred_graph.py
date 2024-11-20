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
from tqdm import tqdm, trange
import numpy as np
import json
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import *
from models import MultitaskClassifier, MultitaskClassifierRoberta
from optimization import *
from utils import *
from models_graph import NumNetMultitaskClassifierDeBERTaV24, NumNetMultitaskClassifierRoberta4, NumNetMultitaskClassifierDeBERTa4, NumNetMultitaskClassifierBERT4
from pathlib import Path
from utils_gnn import convert_to_features_roberta_graph2_no_label, convert_to_features_bert_graph2_no_label, convert_to_features_deberta_graph2_no_label
from collate_fn_pred import OtrDataset
from collate_fn import GraphDataCollator

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
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
    parser.add_argument("--split",
                        default='dev',
                        type=str,
                        required=True,
                        help="data split")

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
    # parser.add_argument("--eval_ratio",
    #                     default=0.5,
    #                     type=float,
    #                     help="portion of data for evaluation")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=7,
                        help="random seed for initialization")
    parser.add_argument('--dropout_prob',
                        type=float,
                        default=-1,
                        help="dropout prob")
    parser.add_argument("--graph_construction",
                        type=int,
                        default=0)
    parser.add_argument('--use_event_chain',
                        nargs='+',
                        #action="store_true",
                        help="use event chain output")
    parser.add_argument("--use_parser_tag",
                        action="store_true")
    parser.add_argument('--rank_loss_ratio',
                        type=float,
                        default=0,
                        help="event rank loss")
    parser.add_argument('--contrastive_loss_ratio',
                        type=float,
                        default=0,
                        help="0: baseline 1: in-contrast-question 2: in-same-passage")
    parser.add_argument("--gcn_steps",
                        type=int,
                        default=4)
    parser.add_argument("--event_loss_ratio",
                        type=int,
                        default=0)
    parser.add_argument('--num_labels',
                        type=int,
                        default=2,
                        help="num_labels")
    parser.add_argument('--n_negs',
                        type=int,
                        default=0,
                        help="number of negatives when negative sampling")
    parser.add_argument("--wo_events",
                        action="store_true")
    parser.add_argument("--orig_optim",
                        action="store_true")
    parser.add_argument("--aftercont",
                        action="store_true")
    parser.add_argument("--deliberate",
                        type = int,
                        default = 0)
    parser.add_argument("--residual_connection",
                        action="store_true")
    parser.add_argument("--question_concat",
                        action="store_true")
    parser.add_argument("--debug",
                        action="store_true")
    parser.add_argument("--use_gcn",
                        action='store_true',
                        help="Whether to use gcn.")
    parser.add_argument('--abl',
                        type=int,
                        default=0,
                        help="ablation level")
    parser.add_argument("--deliberate_ffn",
                        type = int,
                        default=2048,
                        help="deliberate ffn size")
    parser.add_argument("--max_question_length",
                        default=35,
                        type=int,
                        help="The maximum total question word length after spacy tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    print(args)
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))

    random.seed(args.seed)
    np.random.seed(args.seed)
    if is_torch_available():
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    task_name = args.task_name.lower()

    logger.info("current task is " + str(task_name))

    model_state_dict = torch.load(args.model_dir + "pytorch_model.bin")

    if 'roberta' in args.model:
        tokenizer = RobertaTokenizer.from_pretrained(args.model)
        cache_dir = PYTORCH_PRETRAINED_ROBERTA_CACHE / 'distributed_-1'
        if args.use_gcn:
            if args.deliberate:
                print("roberta model")
                model = NumNetMultitaskClassifierRoberta4.from_pretrained(args.model, state_dict = model_state_dict, cache_dir=cache_dir, mlp_hid=args.mlp_hid_size, gcn_steps=args.gcn_steps,
                                                    max_question_length=args.max_question_length, use_gcn=args.use_gcn, event_loss_ratio=args.event_loss_ratio, use_parser_tag=args.use_parser_tag,
                                                    contrastive_loss_ratio=args.contrastive_loss_ratio, dropout_prob=args.dropout_prob, rank_loss_ratio=args.rank_loss_ratio,
                                                    num_labelss=args.num_labels, wo_events = args.wo_events, residual_connection=args.residual_connection, 
                                                    question_concat = args.question_concat, deliberate = args.deliberate, ablation = args.abl, 
                                                    deliberate_ffn=args.deliberate_ffn)
        else:
            model = MultitaskClassifierRoberta.from_pretrained(args.model, state_dict=model_state_dict,
                                                            cache_dir=cache_dir, mlp_hid=args.mlp_hid_size)
    elif 'deberta' in args.model:
        if args.use_gcn:
            if args.deliberate:
                if "-v2" in args.model or "-v3" in args.model:
                    logger.info(f"debertav2")
                    tokenizer = AutoTokenizer.from_pretrained(args.model)
                    model = NumNetMultitaskClassifierDeBERTaV24.from_pretrained(args.model, state_dict=model_state_dict, mlp_hid=args.mlp_hid_size, gcn_steps=args.gcn_steps,
                                                                    max_question_length=args.max_question_length, use_gcn=args.use_gcn, event_loss_ratio=args.event_loss_ratio, use_parser_tag=args.use_parser_tag,
                                                                    contrastive_loss_ratio=args.contrastive_loss_ratio, dropout_prob=args.dropout_prob, rank_loss_ratio=args.rank_loss_ratio,
                                                                    num_labelss=args.num_labels, wo_events = args.wo_events, question_concat = args.question_concat, 
                                                                    residual_connection=args.residual_connection, deliberate = args.deliberate, ablation = args.abl,
                                                                    deliberate_ffn=args.deliberate_ffn)
                else:
                    logger.info("loading deberta model")
                    tokenizer = DebertaTokenizer.from_pretrained(args.model)
                    model = NumNetMultitaskClassifierDeBERTa4.from_pretrained(args.model, state_dict = model_state_dict, mlp_hid=args.mlp_hid_size, gcn_steps=args.gcn_steps,
                                                        max_question_length=args.max_question_length, use_gcn=args.use_gcn, event_loss_ratio=args.event_loss_ratio, use_parser_tag=args.use_parser_tag,
                                                        contrastive_loss_ratio=args.contrastive_loss_ratio, dropout_prob=args.dropout_prob, rank_loss_ratio=args.rank_loss_ratio,
                                                        num_labelss=args.num_labels, wo_events = args.wo_events, residual_connection=args.residual_connection, 
                                                        question_concat = args.question_concat, deliberate = args.deliberate, ablation = args.abl, 
                                                        deliberate_ffn=args.deliberate_ffn)
    else:
        if args.use_gcn:
            if args.deliberate:
                tokenizer = BertTokenizer.from_pretrained(args.model)
                logger.info("loading bert model")
                #cache_dir = PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_-1'
                model = NumNetMultitaskClassifierBERT4.from_pretrained(args.model, state_dict = model_state_dict, mlp_hid=args.mlp_hid_size, gcn_steps=args.gcn_steps,
                                                    max_question_length=args.max_question_length, use_gcn=args.use_gcn, event_loss_ratio=args.event_loss_ratio, use_parser_tag=args.use_parser_tag,
                                                    contrastive_loss_ratio=args.contrastive_loss_ratio, dropout_prob=args.dropout_prob, rank_loss_ratio=args.rank_loss_ratio,
                                                    num_labelss=args.num_labels, wo_events = args.wo_events, residual_connection=args.residual_connection, 
                                                    question_concat = args.question_concat, deliberate = args.deliberate, ablation = args.abl, 
                                                    deliberate_ffn=args.deliberate_ffn)
    model.to(device)
    #hyper_params = args.model_dir.split('/')[1].split('_')
    eval_file = args.split

    print("=" * 50 + "Evaluating %s" % eval_file + "=" * 50)
    eval_data = load_data(args.data_dir, "individual_%s" % eval_file, args.file_suffix)
    if 'roberta' in args.model:
        if args.use_gcn:
            eval_features = convert_to_features_roberta_graph2_no_label(eval_data, tokenizer,
                                                                       max_length=args.max_seq_length,
                                                                       max_question_length=args.max_question_length,
                                                                       evaluation=True,
                                                                       graph_construction=args.graph_construction)
        else:
            eval_features = convert_to_features_roberta_no_label(eval_data, tokenizer,
                                                            max_length=args.max_seq_length, evaluation=True)
    elif 'deberta' in args.model:
        logger.info("loading deberta data")
        if args.use_gcn:
            eval_features = convert_to_features_deberta_graph2_no_label(eval_data, tokenizer,
                                                                       max_length=args.max_seq_length,
                                                                       max_question_length=args.max_question_length,
                                                                       evaluation=True,
                                                                       graph_construction=args.graph_construction)
    else:
        if args.use_gcn:
            logger.info("loading bert data")
            eval_features = convert_to_features_bert_graph2_no_label(eval_data, tokenizer,
                                                                          max_length=args.max_seq_length,
                                                                          max_question_length=args.max_question_length,
                                                                          evaluation=True,
                                                                          graph_construction=args.graph_construction)

    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_features))
    logger.info("  Batch size = %d", args.eval_batch_size)

    #eval_input_ids = torch.tensor(select_field(eval_features, 'input_ids'), dtype=torch.long)
    #eval_input_mask = torch.tensor(select_field(eval_features, 'mask_ids'), dtype=torch.long)
    #eval_segment_ids = torch.tensor(select_field(eval_features, 'segment_ids'), dtype=torch.long)
    eval_offsets = select_field(eval_features, 'offset')
    eval_key_indices = torch.tensor(list(range(len(eval_offsets))), dtype=torch.long)

    # collect unique question ids for EM calculation
    question_ids = select_field(eval_features, 'question_id')
    #eval_data = TensorDataset(eval_input_ids, eval_input_mask, eval_segment_ids, eval_key_indices)
    # Run prediction for full data
    #eval_sampler = SequentialSampler(eval_data)
    #eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
    
    if args.deliberate:
        eval_dataset = OtrDataset(
            eval_features, args.eval_batch_size, evaluation=True)
        eval_graph_data_collator = GraphDataCollator(
            device=device, batch_size=args.eval_batch_size, evaluation=True, contrastive_loss=(args.contrastive_loss_ratio or args.deliberate))
        eval_dataloader = DataLoader(
            eval_dataset, shuffle=False, collate_fn=eval_graph_data_collator, batch_size=1)  # batch_size 1 -> dynamic batch size in dataloader        


    eval_loss, eval_accuracy, best_eval_f1, nb_eval_examples, nb_eval_steps = 0.0, 0.0, 0.0, 0, 0

    all_predictions = {}
    model.eval()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        input_ids, input_masks, segment_ids, instance_indices, head_masks, \
            question_len, edges, q_tr_word_idx, q_event_word_idx, same_p_mask = batch
        offsets, lengths = flatten_answers_no_label([eval_offsets[i]
                                                     for i in instance_indices])

        with torch.no_grad():
            #logits = model(input_ids, offsets, lengths, token_type_ids=segment_ids,
            #               attention_mask=input_masks)
            logits  = model(input_ids, offsets, lengths, edges=edges,
                                        bpe_to_node=head_masks, question_len=question_len, attention_mask=input_masks,
                                        token_type_ids=segment_ids,
                                        q_tr_word_idx=q_tr_word_idx, q_event_word_idx=q_event_word_idx, same_p_mask=same_p_mask)

        logits = logits.detach().cpu().numpy()

        nb_eval_examples += logits.shape[0]
        nb_eval_steps += 1

        batch_preds = np.argmax(logits, axis=1)
        bi = 0
        for l, idx in enumerate(instance_indices):
            pred = [int(batch_preds[bi + li]) for li in range(lengths[l])]
            all_predictions[question_ids[idx]] = pred
            bi += lengths[l]

    #with open(f".json" % (eval_file, hyper_params[-6], args.seed), 'w') as outfile:
    with open(f"{args.model_dir}/test_pred.json", 'w') as outfile:
         json.dump(all_predictions, outfile)

if __name__ == "__main__":
    main()