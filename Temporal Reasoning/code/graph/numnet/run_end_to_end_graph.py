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
from transformers import set_seed
from collate_fn import GraphDataCollator, GraphDataCollatorCont, GraphDataset, OtrGraphDataset, OtrDataset
from models import MultitaskClassifier, MultitaskClassifierRoberta
from models_graph import NumNetMultitaskClassifierDeBERTa4, NumNetMultitaskClassifierDeBERTaV24, NumNetMultitaskClassifierElectra4, NumNetMultitaskClassifierRoberta, NumNetMultitaskClassifierRoberta2, \
    NumNetMultitaskClassifierRoberta4, NumNetMultitaskClassifierBERT4
from models_graph_prev import GraphMultitaskClassifierRoberta, GraphMultitaskClassifierRobertaV2
from utils import *
from utils_gnn import convert_to_features_bert_graph2, convert_to_features_roberta_graph2, convert_to_features_deberta_graph2
from optimization import *
from collections import defaultdict
import sys
from pathlib import Path
from datetime import datetime, date

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
PYTORCH_PRETRAINED_ROBERTA_CACHE = Path(os.getenv('PYTORCH_PRETRAINED_ROBERTA_CACHE',
                                                  Path.home() / '.pytorch_pretrained_roberta'))


def select_neg_samples():
    pass


def main():
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .json files (or other data files) for the task.")
    parser.add_argument("--model_path", default=None, type=str, required=False,
                        help="pre-trained model path. if None, same as model type")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="pre-trained model selected in the list: roberta-base, "
                             "roberta-large, bert-base, bert-large, deberta ")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--log_dir",
                        default="logs",
                        type=str,
                        required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--file_suffix",
                        default=None,
                        type=str,
                        required=True,
                        help="unique identifier for data file")
    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=178,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--max_question_length",
                        default=35,
                        type=int,
                        help="The maximum total question sequence length after spacy tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--instance",
                        type=bool,
                        default=True,
                        help="whether to create sample as instance: 1Q: multiple answers")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--finetune",
                        action='store_true',
                        help="Whether to finetune LM.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--use_qdgat",
                        default=0,
                        type=int,
                        help="Whether to use qdgat.")
    parser.add_argument("--load_model",
                        type=str,
                        help="cosmos_model.bin, te_model.bin",
                        default="")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--mlp_hid_size",
                        default=64,
                        type=int,
                        help="hid dimension for MLP layer.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--bert_learning_rate",
                        default=1e-5,
                        type=float,
                        help="The initial learning rate for BERT.")
    parser.add_argument("--train_ratio",
                        default=1.0,
                        type=float,
                        help="ratio of training samples")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--cuda',
                        type=str,
                        default="",
                        help="cuda index")
    parser.add_argument("--gcn_steps",
                        type=int,
                        default=4)
    parser.add_argument("--event_loss_ratio",
                        type=int,
                        default=0)
    parser.add_argument("--use_parser_tag",
                        action="store_true")
    parser.add_argument('--contrastive_loss_ratio',
                        type=float,
                        default=0,
                        help="0: baseline 1: in-contrast-question 2: in-same-passage")
    parser.add_argument('--dropout_prob',
                        type=float,
                        default=-1,
                        help="dropout prob")
    parser.add_argument("--use_gcn",
                        action='store_true',
                        help="Whether to use gcn.")
    parser.add_argument("--graph_construction",
                        type=int,
                        default=0)
    parser.add_argument('--use_event_chain',
                        nargs='+',
                        #action="store_true",
                        help="use event chain output")
    parser.add_argument('--rank_loss_ratio',
                        type=float,
                        default=0,
                        help="event rank loss")
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
    parser.add_argument("--deliberate",
                        type = int,
                        default = 0,
                        help="number of transformer layers to deliberate")
    parser.add_argument("--deliberate_ffn",
                        type = int,
                        default = 2048,
                        help="hidden size in deliberation layer")
    parser.add_argument("--residual_connection",
                        action="store_true")
    parser.add_argument("--question_concat",
                        action="store_true")
    parser.add_argument("--debug",
                        action="store_true")
    parser.add_argument("--share",
                        action="store_true",
                        help="share answer prediction module")
    parser.add_argument("--abl",
                        type = int,
                        default = 0,
                        help="0: no ablation, 1: remove batch attention 2: remove seq attention")

    args = parser.parse_args()
    #for k,v in args.__dict__.items():
    #    print(f"\"{k}\",\"{v}\",")

    args.output_dir = args.output_dir + "/"
    if args.use_gcn:
        args.output_dir +=f"/gcn{args.use_gcn}"
    if args.contrastive_loss_ratio > 0:
        args.output_dir += f"_cl{args.contrastive_loss_ratio}"
    if args.n_negs > 0:
        args.output_dir += f"_neg{args.n_negs}"
    if args.wo_events:
        args.output_dir += f"_woevent"
    if args.rank_loss_ratio > 0:
        args.output_dir += f"_erl{args.rank_loss_ratio}"
    if args.orig_optim:
        args.output_dir += f"_origopt"
    if args.deliberate:
        args.output_dir += f"_deliberate{args.deliberate}{args.deliberate_ffn}"
        if args.residual_connection:
            args.output_dir += f"_rescon"
    if args.question_concat:
        args.output_dir += f"_qconcat"
    if args.abl:
        args.output_dir +=f"/abl{args.abl}/"
    if args.share:
        args.output_dir +=f"/share/"
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.load_model:
        if not args.debug:
            raise ValueError(
                "Output directory ({}) already exists and is not empty.".format(args.output_dir))

    os.makedirs(args.output_dir, exist_ok=True)

    args.log_dir = args.output_dir
    fileHandler = logging.FileHandler(
        f"{args.log_dir}/log_train.txt")
    logger.addHandler(fileHandler)
    logger.info("args: {}".format(args))
    now = datetime.now()
    today = date.today()
    d4 = today.strftime("%b-%d-%Y")
    current_time = now.strftime("%H:%M:%S")
    logger.info(f"time: {d4} {current_time}")
    logger.info(f"run_end_to_end_graph.py")

    if args.use_gcn:
        assert args.use_qdgat == 0

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))
    print("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      

    args.train_batch_size = int(
        args.train_batch_size / args.gradient_accumulation_steps)
    # fix all random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    if is_torch_available():
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    # https://github.com/pytorch/pytorch/issues/16894
    torch.set_num_threads(8)
    os.environ['OMP_NUM_THREADS'] = '1'

    if not args.do_train and not args.do_eval:
        raise ValueError(
            "At least one of `do_train` or `do_eval` must be True.")

    task_name = args.task_name.lower()
    logger.info("current task is " + str(task_name))
    
    if not args.model_path:
        args.model_path = args.model_type
    # construct model
    if 'roberta' in args.model_type:
        tokenizer = RobertaTokenizer.from_pretrained(args.model_path)
        cache_dir = f"{PYTORCH_PRETRAINED_ROBERTA_CACHE}/distributed_{args.local_rank}"
        if args.deliberate:
            logger.info(f"numnet4")
            model = NumNetMultitaskClassifierRoberta4.from_pretrained(args.model_path, cache_dir=cache_dir, mlp_hid=args.mlp_hid_size, gcn_steps=args.gcn_steps,
                                                            max_question_length=args.max_question_length, use_gcn=args.use_gcn, event_loss_ratio=args.event_loss_ratio, use_parser_tag=args.use_parser_tag,
                                                            contrastive_loss_ratio=args.contrastive_loss_ratio, dropout_prob=args.dropout_prob, rank_loss_ratio=args.rank_loss_ratio,
                                                            num_labelss=args.num_labels, wo_events = args.wo_events, residual_connection=args.residual_connection, question_concat = args.question_concat, deliberate = args.deliberate, ablation = args.abl, share = args.share,
                                                            deliberate_ffn=args.deliberate_ffn)
        else:
            logger.info(f"numnet")
            model = NumNetMultitaskClassifierRoberta.from_pretrained(args.model_path, cache_dir=cache_dir, mlp_hid=args.mlp_hid_size, gcn_steps=args.gcn_steps,
                                                                    max_question_length=args.max_question_length, use_gcn=args.use_gcn, event_loss_ratio=args.event_loss_ratio, use_parser_tag=args.use_parser_tag,
                                                                    contrastive_loss_ratio=args.contrastive_loss_ratio, dropout_prob=args.dropout_prob, rank_loss_ratio=args.rank_loss_ratio,
                                                                    num_labelss=args.num_labels, wo_events = args.wo_events, question_concat = args.question_concat)

    elif 'deberta' in args.model_type:
        if "-v2" in args.model_path or "-v3" in args.model_path:
            logger.info(f"debertav2")
            tokenizer = AutoTokenizer.from_pretrained(args.model_path)
            model = NumNetMultitaskClassifierDeBERTaV24.from_pretrained(args.model_path, mlp_hid=args.mlp_hid_size, gcn_steps=args.gcn_steps,
                                                                    max_question_length=args.max_question_length, use_gcn=args.use_gcn, event_loss_ratio=args.event_loss_ratio, use_parser_tag=args.use_parser_tag,
                                                                    contrastive_loss_ratio=args.contrastive_loss_ratio, dropout_prob=args.dropout_prob, rank_loss_ratio=args.rank_loss_ratio,
                                                                    num_labelss=args.num_labels, wo_events = args.wo_events, question_concat = args.question_concat, 
                                                                    residual_connection=args.residual_connection, deliberate = args.deliberate, ablation = args.abl, share = args.share,
                                                                    deliberate_ffn=args.deliberate_ffn)
        else:   
            tokenizer = DebertaTokenizer.from_pretrained(args.model_path)
        # if args.deliberate:
            logger.info(f"deberta")
            model = NumNetMultitaskClassifierDeBERTa4.from_pretrained(args.model_path, mlp_hid=args.mlp_hid_size, gcn_steps=args.gcn_steps,
                                                                    max_question_length=args.max_question_length, use_gcn=args.use_gcn, event_loss_ratio=args.event_loss_ratio, use_parser_tag=args.use_parser_tag,
                                                                    contrastive_loss_ratio=args.contrastive_loss_ratio, dropout_prob=args.dropout_prob, rank_loss_ratio=args.rank_loss_ratio,
                                                                    num_labelss=args.num_labels, wo_events = args.wo_events, question_concat = args.question_concat, 
                                                                    residual_connection=args.residual_connection, deliberate = args.deliberate, ablation = args.abl, share = args.share,
                                                                    deliberate_ffn=args.deliberate_ffn)
    elif "electra" in args.model_type:
        tokenizer = ElectraTokenizer.from_pretrained(args.model_path)
        model = NumNetMultitaskClassifierElectra4.from_pretrained(args.model_path, mlp_hid=args.mlp_hid_size, gcn_steps=args.gcn_steps,
                                                                max_question_length=args.max_question_length, use_gcn=args.use_gcn, event_loss_ratio=args.event_loss_ratio, use_parser_tag=args.use_parser_tag,
                                                                contrastive_loss_ratio=args.contrastive_loss_ratio, dropout_prob=args.dropout_prob, rank_loss_ratio=args.rank_loss_ratio,
                                                                num_labelss=args.num_labels, wo_events = args.wo_events, question_concat = args.question_concat,
                                                                residual_connection=args.residual_connection, deliberate = args.deliberate, ablation = args.abl, share = args.share,
                                                                deliberate_ffn=args.deliberate_ffn)
    else:  # not use this
        tokenizer = BertTokenizer.from_pretrained(args.model_path)
        logger.info(f"bert")
        model = NumNetMultitaskClassifierBERT4.from_pretrained(args.model_path, mlp_hid=args.mlp_hid_size, gcn_steps=args.gcn_steps,
                                                                max_question_length=args.max_question_length, use_gcn=args.use_gcn, event_loss_ratio=args.event_loss_ratio, use_parser_tag=args.use_parser_tag,
                                                                contrastive_loss_ratio=args.contrastive_loss_ratio, dropout_prob=args.dropout_prob, rank_loss_ratio=args.rank_loss_ratio,
                                                                num_labelss=args.num_labels, wo_events = args.wo_events, question_concat = args.question_concat, deliberate = args.deliberate, 
                                                                residual_connection=args.residual_connection, 
                                                                ablation = args.abl, share = args.share,
                                                                deliberate_ffn=args.deliberate_ffn)
    model.to(device)
    if args.do_train:
        train_data = load_data(args.data_dir, "train",
                               args.file_suffix, args.train_ratio) # 24523
        if 'roberta' in args.model_type:
            train_features = convert_to_features_roberta_graph2(train_data, tokenizer, instance=False,
                                                                max_length=args.max_seq_length, max_question_length=args.max_question_length,
                                                                end_to_end=True, is_qdgat=args.use_qdgat,
                                                                graph_construction=args.graph_construction,
                                                                event_chain_files=args.use_event_chain, train_ratio = args.train_ratio)#,
                                                               #mask_what = bool(args.rank_loss_ratio))
        elif 'deberta' in args.model_type:
            train_features = convert_to_features_deberta_graph2(train_data, tokenizer, instance=False,
                                                                max_length=args.max_seq_length, max_question_length=args.max_question_length,
                                                                end_to_end=True, is_qdgat=args.use_qdgat,
                                                                graph_construction=args.graph_construction,
                                                                event_chain_files=args.use_event_chain, train_ratio = args.train_ratio)#,
                                                               #mask_what = bool(args.rank_loss_ratio))
        else:
            train_features = convert_to_features_bert_graph2(train_data, tokenizer, instance=False,
                                                                max_length=args.max_seq_length, max_question_length=args.max_question_length,
                                                                end_to_end=True, graph_construction=args.graph_construction, train_ratio = args.train_ratio)

        num_train_steps = int(
            len(train_features) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)
        logger.info(f"# train instances: {len(train_features)}")
        all_offsets = select_field(train_features, 'offset')
        all_labels = select_field(train_features, 'label')
        all_events = select_field(train_features, 'events')
        all_key_indices = list(range(len(all_labels)))

        if (args.contrastive_loss_ratio or args.deliberate ) :
            train_dataset = OtrDataset(
                train_features, args.train_batch_size, train_ratio = args.train_ratio)
            graph_data_collator = GraphDataCollator(
                device=device, batch_size=args.train_batch_size, evaluation=False, contrastive_loss=(args.contrastive_loss_ratio or args.deliberate) )
            train_dataloader = DataLoader(
                train_dataset, shuffle=True, collate_fn=graph_data_collator, batch_size=1)  # batch_size 1 -> dynamic batch size in dataloader
        else:
            train_dataset = GraphDataset(
                train_features, all_key_indices)
            graph_data_collator = GraphDataCollator(device=device)
            train_dataloader = DataLoader(
                train_dataset, shuffle=True, collate_fn=graph_data_collator, batch_size=args.train_batch_size)

        model.train()

        model_to_save = model.module if hasattr(
            model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")

        # Prepare optimizer
        # https://github.com/llamazing/numnet_plus/blob/43928b2acd02f5a494688ffcd1d3da6e661da5d3/tools/model.py#L7
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        if args.orig_optim:
            if 'roberta' in args.model_type:
                startswith = 'roberta'
            elif 'deberta' in args.model_type:
                startswith = 'deberta'
            else:
                startswith = 'bert'
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if (not n.startswith(
                    startswith) and not any(nd in n for nd in no_decay))], 'weight_decay': 0.01, 'lr': args.learning_rate},
                {'params': [p for n, p in param_optimizer if (not n.startswith(
                    startswith) and any(nd in n for nd in no_decay))], 'weight_decay': 0.0, 'lr': args.learning_rate},
                {'params': [p for n, p in param_optimizer if n.startswith(
                    startswith) and not any(nd in n for nd in no_decay)], 'weight_decay': 0.01, 'lr': args.bert_learning_rate},
                {'params': [p for n, p in param_optimizer if n.startswith(
                    startswith) and any(nd in n for nd in no_decay)], 'weight_decay': 0, 'lr': args.bert_learning_rate},
            ] 
        else:
            if 'roberta' in args.model_type:
                startswith = 'roberta'
            elif 'deberta' in args.model_type:
                startswith = 'deberta'
            else:
                startswith = 'bert'
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if (not n.startswith(
                    startswith) and not any(nd in n for nd in no_decay))], 'weight_decay': 0.01},
                {'params': [p for n, p in param_optimizer if (not n.startswith(
                    startswith) and any(nd in n for nd in no_decay))], 'weight_decay': 0.0},
                {'params': [p for n, p in param_optimizer if n.startswith(startswith) and not any(
                    nd in n for nd in no_decay)], 'weight_decay': 0.01, 'lr': args.bert_learning_rate},
            ]

        t_total = num_train_steps
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=t_total)

        if n_gpu > 1:
            model = torch.nn.DataParallel(model)

        label_map = {0: 'Negative', 1: 'Positive'}

        global_step = 0
        nb_tr_steps = 0
        tr_loss = 0
        tr_acc = 0.0

        best_eval_accuracy = 0.0
        best_eval = 0.3292 
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss, tr_acc = 0.0, 0.0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                input_ids, input_masks, segment_ids, instance_indices, head_masks, question_len, edges, \
                    q_tr_word_idx, q_event_word_idx, same_p_mask = batch
                offsets, labels, lengths = flatten_answers([(all_labels[i], all_offsets[i])
                                                            for i in instance_indices])
                _, events, _ = flatten_answers([(all_events[i], all_offsets[i])
                                                for i in instance_indices])
                labels = torch.tensor(labels).to(device)
                events = torch.tensor(events).to(device)
                logits, loss = model(input_ids, offsets, lengths, edges=edges,
                                    bpe_to_node=head_masks, question_len=question_len, attention_mask=input_masks,
                                    token_type_ids=segment_ids, labels=labels, events=events,
                                    q_tr_word_idx=q_tr_word_idx, q_event_word_idx=q_event_word_idx, same_p_mask=same_p_mask)
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                loss.backward()
                tr_loss += loss.item()
                nb_tr_examples += labels.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # modify learning rate with special warm up BERT uses
                    lr_this_step = args.learning_rate * warmup_linear(global_step / t_total,
                                                                    args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                logits = logits.detach().cpu().numpy()
                labels = labels.to('cpu').numpy()
                if nb_tr_examples % 10000 == 0:
                    logger.info("current train loss is %s" %
                                (tr_loss / float(nb_tr_steps)))
            if args.do_eval:
                # fix dev data to be "individual_dev_end2end_final"
                eval_data = load_data(
                    args.data_dir, "individual_dev", args.file_suffix, args.train_ratio)
                if 'roberta' in args.model_type:
                    eval_features = convert_to_features_roberta_graph2(eval_data, tokenizer, instance=False,
                                                                       max_length=args.max_seq_length, end_to_end=True,
                                                                       max_question_length=args.max_question_length,
                                                                       evaluation=args.do_eval, is_qdgat=args.use_qdgat,
                                                                       graph_construction=args.graph_construction,
                                                                       event_chain_files=args.use_event_chain,
                                                                       mask_what = bool(args.rank_loss_ratio))
                elif 'deberta' in args.model_type:
                    eval_features = convert_to_features_deberta_graph2(eval_data, tokenizer, instance=False,
                                                                       max_length=args.max_seq_length, end_to_end=True,
                                                                       max_question_length=args.max_question_length,
                                                                       evaluation=args.do_eval, is_qdgat=args.use_qdgat,
                                                                       graph_construction=args.graph_construction,
                                                                       event_chain_files=args.use_event_chain,
                                                                       mask_what = bool(args.rank_loss_ratio))
                else:
                    eval_features = convert_to_features_bert_graph2(eval_data, tokenizer, instance=False,
                                                                       max_length=args.max_seq_length, end_to_end=True,
                                                                       max_question_length=args.max_question_length,
                                                                       evaluation=args.do_eval,
                                                                       graph_construction=args.graph_construction)

                logger.info("***** Running evaluation *****")
                logger.info("  Num examples = %d", len(eval_features))

                eval_offsets = select_field(eval_features, 'offset')
                eval_labels = select_field(eval_features, 'label')
                eval_key_indices = list(range(len(eval_labels)))
                eval_events = select_field(eval_features, 'events')

                # collect unique question ids for EM calculation
                question_ids = select_field(eval_features, 'question_id')
                # flatten question_ids
                question_ids = [q for i, q in enumerate(
                    question_ids) for x in range(len(eval_labels[i]))]
                question_cluster = select_field(eval_features, 'question_cluster')
                question_cluster_size = select_field(eval_features, 'cluster_size')
                eval_idv_answers = select_field(
                    eval_features, 'individual_answers')

                if (args.contrastive_loss_ratio or args.deliberate) and not args.n_negs: #dynamic batch
                    eval_dataset = OtrDataset(
                        eval_features, args.train_batch_size, evaluation=True)
                    eval_graph_data_collator = GraphDataCollator(
                        device=device, batch_size=args.train_batch_size, evaluation=True, contrastive_loss=(args.contrastive_loss_ratio or args.deliberate))
                    eval_dataloader = DataLoader(
                        eval_dataset, shuffle=False, collate_fn=eval_graph_data_collator, batch_size=1)  # batch_size 1 -> dynamic batch size in dataloader
                eval_loss, eval_accuracy, nb_eval_examples, nb_eval_steps = 0.0, 0.0, 0, 0
                all_preds, all_golds, max_f1s = [], [], []
                em_counter = 0
                f1_cluster_80 = {}
                model.eval()
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
                                                    bpe_to_node=head_masks, question_len=question_len, attention_mask=input_masks,
                                                    token_type_ids=segment_ids, labels=labels, events=events,
                                                    q_tr_word_idx=q_tr_word_idx, q_event_word_idx=q_event_word_idx, same_p_mask=same_p_mask)

                        logits = logits.detach().cpu().numpy()
                        labels = labels.to('cpu').numpy()
                        tmp_eval_accuracy = accuracy(logits, labels)

                        eval_loss += tmp_eval_loss.mean().item()
                        eval_accuracy += tmp_eval_accuracy

                        nb_eval_examples += labels.shape[0]
                        nb_eval_steps += 1

                        batch_preds = np.argmax(logits, axis=1)
                        bi = 0
                        for l, idx in enumerate(instance_indices):
                            pred = [batch_preds[bi + li] for li in range(lengths[l])]
                            pred_names = [label_map[p] for p in pred]
                            # for relaxed
                            max_f1 = 0
                            instance_matched = 0
                            for gold in eval_idv_answers[idx]:
                                label_names = [label_map[l] for l in gold]
                                if pred_names == label_names: instance_matched = 1
                                if sum(gold) == 0 and sum(pred) == 0:  # 이 부분 계산 안하니까 +cl의 성능이 올랐다?
                                    f1 = 1.0
                                else:
                                    f1 = cal_f1(pred_names, label_names, {
                                        v: k for k, v in label_map.items()})
                                if f1 > max_f1:
                                    max_f1 = f1
                                    
                            if question_cluster_size[idx] > 1:
                                if question_cluster[idx] not in f1_cluster_80:
                                    f1_cluster_80[question_cluster[idx]] = 1
                                if max_f1 < 0.8: 
                                    f1_cluster_80[question_cluster[idx]] = 0

                            bi += lengths[l]
                            max_f1s.append(max_f1)
                            em_counter += instance_matched
                        
                        all_preds.extend(batch_preds)

                assert len(all_preds) == len(question_ids)

                f1_cluster_80_res = sum(f1_cluster_80.values()) / len(f1_cluster_80)

                em = exact_match(question_ids, all_golds, all_preds)
                eval_accuracy = eval_accuracy / nb_eval_examples
                label_names = [label_map[l] for l in all_golds]
                pred_names = [label_map[p] for p in all_preds]
                eval_pos_f1 = cal_f1(pred_names, label_names, {
                                     v: k for k, v in label_map.items()})

                logger.info("the current eval accuracy is: %.4f" %
                            eval_accuracy)
                logger.info("the current eval exact match ratio is: %.4f" % em)
                logger.info(
                    "the current eval macro positive class F1 is: %.4f" % np.mean(max_f1s))

                logger.info("the current eval exact match ratio (Relaxed) is: %.4f" % (em_counter / len(eval_features)))
                #logger.info("%d Clusters" % len(em_cluster_relaxed))
                logger.info("the current eval clusrered F1 (max>=0.8) is: %.4f" % (f1_cluster_80_res))

                sum_value = (np.mean(max_f1s) + em_counter / len(eval_features) + f1_cluster_80_res)/3
                if sum_value > best_eval:
                    logger.info("Save at Epoch %s" % epoch)
                    best_eval = sum_value
                    # TODO: just save model
                    torch.save(model_to_save.state_dict(),
                                output_model_file)
                    tokenizer.save_pretrained(args.output_dir)
                model.train()


if __name__ == "__main__":
    main()
