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
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import *
from models import TEClassifierRoberta, TEClassifierRobertaGraph
from te.utils_te import *
from optimization import *
from pathlib import Path
from datetime import date, datetime
from sklearn.metrics import precision_score, recall_score, f1_score

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
PYTORCH_PRETRAINED_ROBERTA_CACHE = Path(os.getenv('PYTORCH_PRETRAINED_ROBERTA_CACHE',
                                               Path.home() / '.pytorch_pretrained_roberta'))
PYTORCH_PRETRAINED_BERT_CACHE = Path(os.getenv('PYTORCH_PRETRAINED_BERT_CACHE',
                                               Path.home() / '.pytorch_pretrained_bert'))

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
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--model_dir",
                        default=None,
                        type=str,
                        required=False,
                        help="The model directory for pretrained QA model")
    parser.add_argument("--te_type",
                        default=None,
                        type=str,
                        required=True,
                        help="subfolder contains TE data")
    ## Other parameters                                                                                            
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--max_question_length",
                        default=8,
                        type=int,
                        help="The maximum total question sequence length after spacy tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--file_suffix",
                        default=None,
                        type=str,
                        help="unique identifier for data file")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--finetune",
                        action='store_true',
                        help="Whether to finetune LM.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
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
    parser.add_argument("--learning_rate_te",
                        default=5e-4,
                        type=float,
                        help="The initial learning rate for Adam TE task.")
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
    parser.add_argument('--device_num',
                        type=str,
                        default="0",
                        help="cuda device number")
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
    for k,v in args.__dict__.items():
        print(f"\"--{k}\",\"{v}\",")
    #os.environ["CUDA_VISIBLE_DEVICES"] = args.device_num

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
    if args.aftercont:
        args.output_dir += f"_aftercont"
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

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs                      
        torch.distributed.init_process_group(backend='nccl')

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    # fix all random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
        #torch.backends.cudnn.deterministic=True
        
    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.load_model:
        pass
        #raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))

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
    logger.info(f"run_te_graph.py")

    task_name = args.task_name.lower()
    logger.info("current task is " + str(task_name))


    if args.te_type in ['tbd']:
        label_map = tbd_label_map
    if args.te_type in ['matres']:
        label_map = matres_label_map
    if args.te_type in ['red']:
        label_map = red_label_map
    num_classes = len(label_map)
        
    # construct model
    if 'roberta' in args.model:
        tokenizer = RobertaTokenizer.from_pretrained(args.model, do_lower_case=args.do_lower_case)
        cache_dir = PYTORCH_PRETRAINED_ROBERTA_CACHE / 'distributed_{}'.format(args.local_rank)
        if "no_transfer" in args.task_name:
            if args.deliberate:
                model = TEClassifierRobertaGraph.from_pretrained(args.model, cache_dir=cache_dir,
                                                             mlp_hid=args.mlp_hid_size, num_classes=num_classes,
                                                             gcn_steps = args.gcn_steps,
                                                             max_question_length=args.max_question_length, 
                                                             use_gcn=args.use_gcn, 
                                                             dropout_prob=args.dropout_prob, 
                                                             residual_connection=args.residual_connection,
                                                             deliberate=args.deliberate,
                                                             deliberate_ffn=args.deliberate_ffn,
                                                             )
            else:
                model = TEClassifierRoberta.from_pretrained(args.model, cache_dir=cache_dir,
                                                        mlp_hid=args.mlp_hid_size, num_classes=num_classes)
        else:
            model_state_dict = torch.load(args.model_dir)
            logger.info(args.model_dir)
            model_state_dict = {k:v for k,v in model_state_dict.items() if "_te" not in k}
            model = TEClassifierRoberta.from_pretrained(args.model, state_dict=model_state_dict, cache_dir=cache_dir,
                                                       mlp_hid=args.mlp_hid_size, num_classes=num_classes)
            del model_state_dict
    else:
        pass
        # tokenizer = BertTokenizer.from_pretrained(args.model, do_lower_case=args.do_lower_case)
        # cache_dir = PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(args.local_rank)

        # if "no_transfer" in args.task_name:
        #     model = TEClassifier.from_pretrained(args.model, cache_dir=cache_dir,
        #                                          mlp_hid=args.mlp_hid_size, num_classes=num_classes)
        # else:
        #     model_state_dict = torch.load(args.model_dir)
        #     logger.info(args.model_dir)
        #     model_state_dict = {k:v for k,v in model_state_dict.items() if "_te" not in k}
        #     model = TEClassifier.from_pretrained(args.model, state_dict=model_state_dict, cache_dir=cache_dir,
        #                                          mlp_hid=args.mlp_hid_size, num_classes=num_classes)
        #     del model_state_dict

    model.to(device)

    if args.do_train:
        if args.te_type in ["matres"]:
            trainIds, devIds = get_train_dev_ids(args.data_dir, args.te_type)
            train_features_te = convert_examples_to_features_te_matres(args.data_dir, args.te_type, 'train',
                                                                tokenizer, args.max_seq_length, True, trainIds)
            eval_features_te = convert_examples_to_features_te_matres(args.data_dir, args.te_type, 'dev',
                                                               tokenizer, args.max_seq_length, True, devIds)
        else:
            train_features_te = convert_examples_to_features_te(args.data_dir, args.te_type, 'train',
                                                                tokenizer, args.max_seq_length, True, max_question_length=args.max_question_length)
            eval_features_te = convert_examples_to_features_te(args.data_dir, args.te_type, 'dev',
                                                               tokenizer, args.max_seq_length, True, max_question_length=args.max_question_length)

        logger.info("***** Running training *****")
        logger.info("  Num TE examples = %d", len(train_features_te))
        logger.info("  Batch size = %d", args.train_batch_size)

        num_train_steps = int(                                                                          
            len(train_features_te) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)
        ## TE data --> tensor                                                                                     
        # all_input_ids_te = torch.tensor(select_field_te(train_features_te, 'input_ids'), dtype=torch.long)
        # all_input_mask_te = torch.tensor(select_field_te(train_features_te, 'input_mask'), dtype=torch.long)
        # all_segment_ids_te = torch.tensor(select_field_te(train_features_te, 'segment_ids'), dtype=torch.long)
        # all_lidx_s = torch.tensor(select_field_te(train_features_te, 'lidx_s'), dtype=torch.long)
        # all_lidx_e = torch.tensor(select_field_te(train_features_te, 'lidx_e'), dtype=torch.long)
        # all_ridx_s = torch.tensor(select_field_te(train_features_te, 'ridx_s'), dtype=torch.long)
        # all_ridx_e = torch.tensor(select_field_te(train_features_te, 'ridx_e'), dtype=torch.long)
        # all_pred_inds = torch.tensor(select_field_te(train_features_te, 'pred_ind'), dtype=torch.long)
        # all_label_te = torch.tensor([f.label for f in train_features_te], dtype=torch.long)
        # all_input_length_te = torch.tensor([f.length for f in train_features_te], dtype=torch.long)
        # all_sample_counters = torch.tensor(select_field_te(train_features_te, 'sample_counter'), dtype=torch.long)
        # ## combine
        # train_data = TensorDataset(all_input_ids_te, all_input_mask_te, all_segment_ids_te, all_label_te,
        #                            all_lidx_s, all_lidx_e, all_ridx_s, all_ridx_e, all_pred_inds,
        #                            all_input_length_te, all_sample_counters)
        
        # logger.info(" Used data = %d", len(train_data))
        
        # if args.local_rank == -1:
        #     tr_limit = int(len(train_data) * args.train_ratio)
        #     selected = random.sample(list(range(len(train_data))), k=tr_limit)
        #     train_sampler = SubsetRandomSampler(selected)
        # else:
        #     train_sampler = DistributedSampler(train_data)

        # train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        train_dataset = TeContDataset(train_features_te, args.train_batch_size, evaluation = False)
        train_data_collator = GraphDataCollatorTE(device = device, batch_size = args.train_batch_size, evaluation = False)
        train_dataloader = DataLoader(
                    train_dataset, shuffle=True, collate_fn=train_data_collator, batch_size=1)

        model.train()

        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")

        # Prepare optimizer
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        te_params = ['linear1_te', 'linear2_te']
        #if args.orig_optim:
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if (not n.startswith(
                    'roberta')) and (not any(nd in n for nd in no_decay)) and
                        (not any(te in n for te in te_params))], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if (n.startswith(
                    'roberta')) and (not any(nd in n for nd in no_decay)) and
                        (not any(te in n for te in te_params))], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if (not n.startswith(
                    'roberta')) and any(nd in n for nd in no_decay) and
                        (not any(te in n for te in te_params))], 'weight_decay': 0.0},
            {'params': [p for n, p in param_optimizer if any(te in n for te in te_params)],
            'lr': args.learning_rate_te, 'weight_decay': 0.0}
        ]
    
        t_total = num_train_steps

        if args.fp16:
            try:
                from apex.optimizers import FusedAdam
                from apex import amp
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
            optimizer = FusedAdam(optimizer_grouped_parameters,
                                  lr=args.learning_rate,
                                  bias_correction=False)
            model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
        else:
            optimizer = BertAdam(optimizer_grouped_parameters,
                                 lr=args.learning_rate,
                                 warmup=args.warmup_proportion,
                                 t_total=t_total)

        if n_gpu > 1:
            model = torch.nn.DataParallel(model)

        global_step = 0

        logger.info("  Total steps = %d", len(train_dataloader))
        best_eval_f1 = 0.0
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss, tr_acc, tr_acc_te = 0.0, 0.0, 0.0
            nb_tr_examples, nb_tr_steps = 0, 0
            
            all_counters = []
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                # batch = tuple(t.to(device) for t in batch)
                # input_ids_te, input_mask_te, segment_ids_te, labels_te, \
                # lidx_s, lidx_e, ridx_s, ridx_e, pred_ind, length_te, counters = batch
                input_ids_te, input_mask_te, segment_ids_te, \
                bpe_to_node, question_len, edges, lidx_s, ridx_s, offsets, \
                labels_te, counters = batch
                lengths = [len(offset) for offset in offsets]
                offsets = [elem for offset in offsets for elem in offset]
                
                all_counters.extend(counters)
                
                loss_te, logit_te = model(input_ids_te, token_type_ids_te=segment_ids_te,
                                          attention_mask_te=input_mask_te, lidx_s=lidx_s, #lidx_e=lidx_e,
                                          ridx_s=ridx_s, #ridx_e=ridx_e, 
                                          # length_te=length_te,
                                          labels_te=labels_te,
                                          lengths = lengths, bpe_to_node=bpe_to_node,
                                          edges = edges, question_len = question_len, offsets = offsets ) # added
                loss = loss_te
                
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                
                tr_loss += loss.item()
                nb_tr_steps += 1
                nb_tr_examples += labels_te.size(0)
                
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # modify learning rate with special warm up BERT uses
                    if args.fp16:
                        lr_this_step = args.learning_rate * warmup_linear(global_step / t_total,
                                                                          args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                logit_te = logit_te.detach().cpu().numpy()
                label_te = labels_te.to('cpu').numpy()
                tr_acc_te += accuracy(logit_te, label_te)
                if nb_tr_steps % 1000 == 0:
                    logger.info("current train loss is: " + str(tr_loss / float(nb_tr_steps)))
                    logger.info("TE training step: " + str(nb_tr_steps))
                    logger.info("TE training samples: " + str(nb_tr_examples))
            # if epoch == 0:
            #     sorted_counters = sorted(all_counters)
            # else:
            #     assert sorted_counters == sorted(all_counters)
                
            model.eval()

            logger.info("***** Running evaluation *****")
            logger.info("  Epoch = %d", epoch)
            logger.info("  Batch size = %d", args.eval_batch_size)
            logger.info("  Num TE examples = %d", len(eval_features_te))
            
            # eval_input_ids_te = torch.tensor(select_field_te(eval_features_te, 'input_ids'), dtype=torch.long)
            # eval_input_mask_te = torch.tensor(select_field_te(eval_features_te, 'input_mask'), dtype=torch.long)
            # eval_segment_ids_te = torch.tensor(select_field_te(eval_features_te, 'segment_ids'), dtype=torch.long)
            # eval_lidx_s = torch.tensor(select_field_te(eval_features_te, 'lidx_s'), dtype=torch.long)
            # eval_lidx_e = torch.tensor(select_field_te(eval_features_te, 'lidx_e'), dtype=torch.long)
            # eval_ridx_s = torch.tensor(select_field_te(eval_features_te, 'ridx_s'), dtype=torch.long)
            # eval_ridx_e = torch.tensor(select_field_te(eval_features_te, 'ridx_e'), dtype=torch.long)
            # eval_pred_inds = torch.tensor(select_field_te(eval_features_te, 'pred_ind'), dtype=torch.long)
            # eval_label_te = torch.tensor([f.label for f in eval_features_te], dtype=torch.long)
            # eval_input_length_te = torch.tensor([f.length for f in eval_features_te], dtype=torch.long)
                
            # eval_data = TensorDataset(eval_input_ids_te, eval_input_mask_te, eval_segment_ids_te, eval_label_te,
            #                           eval_lidx_s, eval_lidx_e, eval_ridx_s, eval_ridx_e, eval_pred_inds,
            #                           eval_input_length_te)
            # # Run prediction for full data
            # eval_sampler = SequentialSampler(eval_data)
            # eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
            eval_dataset = TeContDataset(eval_features_te, args.eval_batch_size, evaluation = True)
            eval_data_collator = GraphDataCollatorTE(device = device, batch_size = args.eval_batch_size, evaluation = True)
            eval_dataloader = DataLoader(
                        eval_dataset, shuffle=False, collate_fn=eval_data_collator, batch_size=1)
            te_preds = []
            
            # for batch in tqdm(eval_dataloader, desc="Evaluating"):
            #     batch = tuple(t.to(device) for t in batch)
            #     input_ids_te, input_mask_te, segment_ids_te, labels_te, \
            #     lidx_s, lidx_e, ridx_s, ridx_e, pred_ind, length_te = batch
                    
            #     with torch.no_grad():
            #         loss_te, logit_te = model(input_ids_te, token_type_ids_te=segment_ids_te,
            #                                   attention_mask_te=input_mask_te, lidx_s=lidx_s, lidx_e=lidx_e,
            #                                   ridx_s=ridx_s, ridx_e=ridx_e, length_te=length_te,
            #                                   labels_te=labels_te)

            #     logit_te = logit_te.detach().cpu().numpy()
            #     pred_te = np.argmax(logit_te, axis=1).tolist()
            #     te_preds.extend(pred_te)
                        
            # idx2label = {k: v for k, v in enumerate(label_map)}
            # te_preds_labels = [idx2label[x] for x in te_preds]
            # te_true_labels = [idx2label[f.label] for f in eval_features_te]
            # eval_f1 = cal_f1(te_preds_labels, te_true_labels, label_map)
            # logger.info("the current eval TE F1 is: " + str(eval_f1))

            all_preds, te_preds, all_golds = [], [], []

            idx2label = {k: v for k, v in enumerate(label_map)}
            #te_true_labels = [idx2label[f.label] for f in eval_features_te]
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                input_ids_te, input_mask_te, segment_ids_te, \
                bpe_to_node, question_len, edges, lidx_s, ridx_s, offsets, \
                labels_te, counters = batch
                lengths = [len(offset) for offset in offsets]
                offsets = [elem for offset in offsets for elem in offset]
                
                with torch.no_grad():
                    loss_te, logit_te = model(input_ids_te, token_type_ids_te=segment_ids_te,
                                          attention_mask_te=input_mask_te, lidx_s=lidx_s, #lidx_e=lidx_e,
                                          ridx_s=ridx_s, #ridx_e=ridx_e, 
                                          # length_te=length_te,
                                          labels_te=labels_te,
                                          lengths = lengths, bpe_to_node=bpe_to_node,
                                          edges = edges, question_len = question_len, offsets = offsets ) # added

                    logit_te = logit_te.detach().cpu().numpy()
                    labels_te = labels_te.detach().cpu().numpy().tolist()
                    pred_te = np.argmax(logit_te, axis=1).tolist()
                    te_preds.extend(pred_te)
                    all_golds.extend(labels_te)

            te_true_labels = [idx2label[x] for x in all_golds]
            te_preds_labels = [idx2label[x] for x in te_preds]
            report = ClassificationReport(args.task_name + "-" + args.te_type, te_true_labels, te_preds_labels)
            logger.info(report)

            # logger.info("f1: ", f1_score(te_true_labels, te_preds_labels, average='micro'))
            # logger.info("precision: ", precision_score(te_true_labels, te_preds_labels, average='micro'))
            # logger.info("recall: ", recall_score(te_true_labels, te_preds_labels, average='micro'))

            if report.rel_f1 > best_eval_f1:
                torch.save(model_to_save.state_dict(), output_model_file)
                best_eval_f1 = report.rel_f1
                logger.info("best eval TE F1 set to: " + str(best_eval_f1))
            model.train()
            
if __name__ == "__main__":
    main()
