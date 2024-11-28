# tt_LAMA_learn_and_forget_test_multigpu.py 
# wloss2_e6_phi_여러 데이터 학습 및 eval.ipynb
# 거의 참고함
import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from TAALM import TAALM, Llama2_kadapter, Llama2_fullfinetune
from torch.nn.parallel import DistributedDataParallel as DDP
from datasets import load_dataset
import argparse
from tqdm import tqdm
import random
from transformers import AutoTokenizer
import pickle 
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from bitsandbytes.optim import AdamW
from utils.RecAdam import RecAdam


parser = argparse.ArgumentParser()
parser.add_argument('--meta_type', type=str, default=None) # lamackl, twiki
parser.add_argument('--gpu_train_batch_size', type=int, default=8) # 보통 4로 함
parser.add_argument('--gpu_review_batch_size', type=int, default=8) # false면 몇개여도 반영 안됨
parser.add_argument('--train_grad_accum', type=bool, default=False)
parser.add_argument('--train_grad_accum_step', type=int, default=1)
parser.add_argument('--train_lr', type=float, default=None)
parser.add_argument('--train_known_masking', type=bool, default=False) ###### 나중에 추가한것. 신경쓰기
parser.add_argument('--fullfinetune_checkpoint', type=str, default="none")

parser.add_argument('--review', type=bool, default=False)
parser.add_argument('--review_data', type=str, default='none')
parser.add_argument('--review_ratio', type=float, default=1.)
parser.add_argument('--review_seed', type=int, default=1)
parser.add_argument('--recadam', type=bool, default=False)
parser.add_argument('--max_epochs', type=int, default=30)
parser.add_argument('--main_test_size', type=int, default=-1) # 0이면 전부 다 하는것
parser.add_argument('--changed_test_size', type=int, default=-1) # 0이면 전부 다 하는것
parser.add_argument('--unchanged_test_size', type=int, default=-1) # 0이면 전부 다 하는것
parser.add_argument('--test_type', type=str, default='none') # finetune, targeted, oracle
parser.add_argument('--model_type', type=str, default='none') # kadapter, qlora
parser.add_argument('--kadapter_checkpoint', type=str, default='none') 
parser.add_argument('--add_to_title', type=str, default='') # kadapter, qlora
parser.add_argument('--theta_adapter_file', type=str, default='none')  
parser.add_argument('--theta_model_name', type=str, default='none') 
parser.add_argument('--phi_adapter_file', type=str, default='none')
parser.add_argument('--phi_model_name', type=str, default='none')
parser.add_argument('--phi_checkpoint', type=str, default='none') 
parser.add_argument('--eval_batch_size', type=int, default=4) 
parser.add_argument('--train_data', type=str, default='none') 
parser.add_argument('--eval_data_changed', type=str, default='none') 
parser.add_argument('--eval_data_unchanged', type=str, default='none') 
parser.add_argument('--save_on_eval', type=bool, default=False) 
parser.add_argument('--task_type', type=str, default='none')  # schematic, descriptive
parser.add_argument('--twiki_num', type=int, default=0) # >0 이면 /temp 에서 checkpoint 줏어서 씀.
parser.add_argument('--twiki_temp_save', type=bool, default=False)
parser.add_argument('--twiki_num_last', type=int, default=100)
parser.add_argument('--skip_light', type=bool, default=False)
parser.add_argument('--token_max_length', type=int, default=-1)
parser.add_argument('--bf16', type=bool, default=False)
parser.add_argument('--rho', type=bool, default=False)
parser.add_argument('--rho_model', type=bool, default=False)
parser.add_argument('--rho_k', type=float, default=0.6)
parser.add_argument('--lm_train', type=bool, default=False)

args = parser.parse_args()






#####################
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12355'

    # 작업 그룹 초기화
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

########## evaluation 관련
def make_task(args, batch):
    if args.meta_type == 'twiki':
        subject_list = batch['subject']
        relation_list= batch['relation']
        object_list= batch['object']
        task_list = []
        for sub, rel, obj in zip(subject_list, relation_list, object_list):
            task = f"Guess the object. \n  subject is {sub} , relation is {rel} , object is {obj}"
            task_list.append(task)
        return task_list
    if args.meta_type == 'lamackl':
        task_list = batch[f'task_{args.task_type}']
        return task_list

# task 의 label_mask 만들기
def giveme_label_mask(query_token, label_token):  
    pallet = torch.zeros(len(query_token))
    for start0 in list(range(len(pallet)- len(label_token)+1))[::-1]:
        if torch.equal(query_token[start0: start0+len(label_token)], label_token):
            pallet[start0: start0+len(label_token)] =1 
            break
    return pallet

## task 라벨마스크 리스트 만들기
def giveme_label_mask_list(task_token, object_target_tokens):
    target_masks = []
    # target_tokens = tokenizer(object_list, add_special_tokens=False).input_ids
    for task, target in zip(task_token.input_ids , object_target_tokens):
        target_mask= giveme_label_mask(task, torch.tensor(target).to('cuda'))
        target_masks.append(target_mask[1:])

    target_masks = torch.stack(target_masks)
    return target_masks
#################################

class Model_wTA(nn.Module):
    def __init__(self, args):
        super(Model_wTA, self).__init__()
        
        if args.model_type == 'kadapter':
            self.theta = Llama2_kadapter.init_theta(model_name=args.theta_model_name, onepiece=True)
            if args.kadapter_checkpoint != 'none':
                with open(args.kadapter_checkpoint, 'rb') as f:
                    temp = pickle.load(f)
                    name_list = []
                    for name, param in self.theta.named_parameters():
                        if param.requires_grad:
                            name_list.append(name)
                    TAALM.inject_param(self.theta, temp['theta_params'], name_list, 'detach_reqgrad' )

        elif args.model_type == 'fullfinetune':
            self.theta= Llama2_fullfinetune.init_theta(model_name=args.theta_model_name, onepiece=True)
            if args.fullfinetune_checkpoint != 'none':
                with open(args.kadapter_checkpoint, 'rb') as f:
                    temp = pickle.load(f)
                    name_list = []
                    for name, param in self.theta.named_parameters():
                        if param.requires_grad:
                            name_list.append(name)
                    TAALM.inject_param(self.theta, temp['theta_params'], name_list, 'detach_reqgrad' )

        elif args.model_type == 'lora':
            self.theta = TAALM.init_theta(regul=True, onepiece=True, adapter_file=args.theta_adapter_file, model_name=args.theta_model_name, lm_train =args.lm_train, qlora_lora='lora')
        elif args.model_type == 'small_qlora':
            self.theta = TAALM.init_theta(regul=True, onepiece=True, adapter_file=args.theta_adapter_file, model_name=args.theta_model_name, lm_train =args.lm_train, qlora_lora='small_qlora')
        else:
            self.theta = TAALM.init_theta(regul=True, onepiece=True, adapter_file=args.theta_adapter_file, model_name=args.theta_model_name, lm_train =args.lm_train)

        
        
        if args.rho_model:
            self.rho = TAALM.init_theta(regul=True, onepiece=True, adapter_file=args.theta_adapter_file, model_name=args.theta_model_name, lm_train =args.lm_train)
            for param in self.rho.parameters():
                param.requires_grad = False

        if args.meta_type == 'twiki' and args.twiki_num > 0:
            #### theta 'temp' checkpoint load =========
            with open(f'./temp/temp_{args.test_type}_{args.model_type}{"_review" if args.review else ""}{"_recadam" if args.recadam else ""}{"_"+ args.add_to_title if args.add_to_title != "" else ""}{"_"+ str(args.twiki_num -1)}.pkl', 'rb') as f:
                history = pickle.load(f)
            theta_init_names=[]
            for name, param in self.theta.named_parameters():
                if param.requires_grad:
                    theta_init_names.append(name)
            theta_param = [param.to('cuda').requires_grad_() for param in history['theta_params']]
            TAALM.inject_param(self.theta, theta_param, theta_init_names, 'assign')
            ###=====================

        if args.test_type=='targeted':
            self.phi = TAALM.init_phi(regul=True, onepiece=True, model_name=args.phi_model_name, adapter_file=args.phi_adapter_file)
            #### phi checkpoint load =========
            with open(args.phi_checkpoint, 'rb') as f:
                history = pickle.load(f)
            phi_init_names=[]
            for name, param in self.phi.named_parameters():
                if param.requires_grad:
                    phi_init_names.append(name)
            phi_param = [param.to('cuda') for param in history['phi_params']]
            TAALM.inject_param(self.phi, phi_param, phi_init_names, 'assign')
            ###=====================

            for name, param in self.phi.named_parameters():
                param.requires_grad = False
            self.phi.eval()
        elif args.test_type=='finetune':
            self.phi = None
        elif args.test_type=='oracle':
            self.phi = None
        

    def forward(self, token, mode0, data=None, train_type=None, batch_data=None, weight_vector=None):# train_data, tokenizer):

        if train_type == None:
            train_type = args.test_type
        
        if mode0=='train':
            self.theta.train()
            if train_type=='targeted':
                with torch.no_grad():
                    # token = tokenizer(train_data, return_tensors='pt', padding=True, truncation=True).to('cuda')
                    output = self.phi(**token)
                    logit0 = output.logits.squeeze(-1)[:,1:]
                    w_action = torch.sigmoid(logit0) * token.attention_mask[:,:-1]
                
                loss, loss_origin, w_loss_sepa, _ = TAALM.calcul_loss(gamma=None, theta=self.theta, inputs=token, tt_weight_vector=w_action, tt_target_weight=True, tt_L2=False,  known_masking=args.train_known_masking, onepiece=True)
            elif train_type=='finetune':
                loss, loss_origin, w_loss_sepa, _ = TAALM.calcul_loss(gamma=None, theta=self.theta, inputs=token, tt_weight_vector=torch.tensor(1.).to('cuda'), tt_target_weight=False, tt_L2=False,  known_masking=False, onepiece=True)

            elif args.test_type=='oracle':
                loss, loss_origin, w_loss_sepa, _ = TAALM.calcul_loss(gamma=None, theta=self.theta, inputs=token, tt_weight_vector=weight_vector.to('cuda'), tt_target_weight=True, tt_L2=False,  known_masking=False, onepiece=True)

            return loss
        
        elif mode0=='rho_train':
            
            loss, loss_origin, w_loss_sepa, nll_loss, log_probs0 = TAALM.calcul_loss(gamma=None, theta=self.theta, inputs=token, tt_weight_vector=torch.tensor(1.).to('cuda'), tt_target_weight=False, tt_L2=False,  known_masking=False, onepiece=True, giveme_logprobs=True)
            
            with torch.no_grad():
                # prob now
                probs = torch.exp(-log_probs0)
                labels0 = torch.clamp(token['input_ids'][:,1:], min=0).unsqueeze(-1)
                probs =probs.gather(dim=-1, index=labels0).squeeze(-1)
                len0 = probs.shape[-1]
                # prob origin
                original_probs =batch_data['original_prob']
                original_probs = torch.stack(original_probs[0]).to('cuda').to(probs.dtype).transpose(1,0)
                original_probs = original_probs[:,-len0:]
                
                probs_change =  probs - original_probs
                rho_weight = probs_change.argsort(dim=-1) / probs_change.shape[-1]
                rho_weight = (rho_weight > args.rho_k).float().to(rho_weight.dtype)

                mask0 = token['attention_mask'][:,:-1]
            rho_weight = rho_weight * mask0
            rho_loss = nll_loss * rho_weight
            # rho_loss = rho_loss.sum(dim=-1) / rho_weight.sum(dim=-1)
            rho_loss = rho_loss.sum(dim=-1) / (mask0.sum(dim=-1) * args.rho_k)
            rho_loss = rho_loss.mean()
            return rho_loss
        
        elif mode0=='rho_model_train':
            self.rho.eval()
            loss, loss_origin, w_loss_sepa, nll_loss, log_probs0 = TAALM.calcul_loss(gamma=None, theta=self.theta, inputs=token, tt_weight_vector=torch.tensor(1.).to('cuda'), tt_target_weight=False, tt_L2=False,  known_masking=False, onepiece=True, giveme_logprobs=True)

            with torch.no_grad():
                outputs = self.rho(**token)
                original_probs = nn.functional.softmax(outputs.logits[:,:-1, :], dim=-1)
            
                # prob now
                probs = torch.exp(-log_probs0)
                labels0 = torch.clamp(token['input_ids'][:,1:], min=0).unsqueeze(-1)
                probs =probs.gather(dim=-1, index=labels0).squeeze(-1)
                len0 = probs.shape[-1]
                # prob origin
                original_probs =original_probs.gather(dim=-1, index=labels0).squeeze(-1)
                
                probs_change =  probs - original_probs
                rho_weight = probs_change.argsort(dim=-1) / probs_change.shape[-1]
                rho_weight = (rho_weight > args.rho_k).float().to(rho_weight.dtype)

                mask0 = token['attention_mask'][:,:-1]
            rho_weight = rho_weight * mask0
            rho_loss = nll_loss * rho_weight
            # rho_loss = rho_loss.sum(dim=-1) / rho_weight.sum(dim=-1)
            rho_loss = rho_loss.sum(dim=-1) / (mask0.sum(dim=-1) * args.rho_k)
            rho_loss = rho_loss.mean()
            return rho_loss

        elif mode0=='eval':
            self.theta.eval()
            with torch.no_grad():
                response=self.theta(**token)
                log_probs = -nn.functional.log_softmax(response.logits[:,:-1,:], dim=-1)
                labels0 = torch.clamp(token['input_ids'][:,1:], min=0).unsqueeze(-1)
                nll_loss = log_probs.gather(dim=-1, index=labels0).squeeze()
                mask0 = token['attention_mask'][:,:-1]
                nll_loss = nll_loss * mask0
                # history['task_p'].append(nll_loss)
                targeted_loss = nll_loss * data['label_masks']
                task_targeted_loss = targeted_loss.sum(dim=-1) / (mask0 * data['label_masks']).sum(dim=-1)

                # task_targeted_loss = task_targeted_loss.mean()
                # history['task_targeted_loss'].append(task_targeted_loss)

                #acc
                output_tk = torch.argmin(log_probs, dim=-1) 
                
                output_labeled = output_tk * data['label_masks']  # ([10, 28])  

                scores = []
                for masked_output, label_token in zip(output_labeled, data['object_target_tokens']):
                    output_cop = masked_output[masked_output !=0] 
                    score = output_cop == torch.tensor(label_token).to('cuda')
                    score = score.float().mean()
                    scores.append(score)

                # scores
                # score = torch.stack(scores).mean()
                # history['accuracy'].append(score)


            self.theta.train()
            return task_targeted_loss, torch.stack(scores)
        
        # elif mode0=='theta_only':
        #     with torch.no_grad():
        #         return self.theta(**token)



def demo_basic(rank, world_size, train_data, eval_data_changed, eval_data_unchanged,  review_data, unchanged_ppl, eval_result): # run()
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)
    torch.cuda.set_device(rank) ##

    if rank==0:
        print(args)

    ## initalize  # rho 때문에 여기로 옮김
    model = Model_wTA(args).to(rank)
    
    model_name = "meta-llama/Llama-2-7b-hf"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    if args.bf16:  # 이거 하려면   gloo -> nccl 로 바꿔야됨
        for name, param in model.named_parameters():
            # print(f"{name}  {'layernorm' in name}")
            if param.dtype == torch.float32:
                param.data = param.data.to(torch.bfloat16)
            # if 'layernorm' in name:
                # param.data == param.data.to(torch.float32)
        
        # for name, param in model.named_modules():
        #     if 'ln' in name:
        #         # print(param)
        #         param.weight.data = param.weight.data.to(torch.float32)
        #         param.bias.data = param.bias.data.to(torch.float32)

    # if rank==0:
    #     for name, param in model.named_parameters():
    #         if param.requires_grad:
    #             print(f"{param.requires_grad} {name}")

    
    #####################
    ## dataset 배분
    if args.rho:
    
        def origin_prob(example):
            model.theta.eval()
            with torch.no_grad():
                if args.meta_type=='twiki':
                    key0 = 'text'
                elif args.meta_type=='lamackl':
                    key0 = 'evidence'
                input0 = tokenizer(example[key0], return_tensors='pt', max_length=args.token_max_length, padding='max_length', truncation=True).to(rank)
                # original = ddp_model(input0, mode0='theta_only')
                original = model.theta(**input0)
                probs = nn.functional.softmax(original.logits[:,:-1,:], dim=-1)
                labels0 = torch.clamp(input0['input_ids'][:,1:], min=0).unsqueeze(-1)
                label_probs = probs.gather(dim=-1, index= labels0).squeeze(-1)
                example['original_prob'] = label_probs.to(torch.float32)#.to('cpu')
            return example
        train_data = train_data.map(origin_prob)
        model.theta.train()

    if train_data.num_rows  % world_size ==0:
        block_size = (train_data.num_rows // world_size) #
    else:
        block_size = (train_data.num_rows // world_size) + 1

    train_sampler = DistributedSampler(train_data, num_replicas=world_size, rank=rank, shuffle=True, seed=42)
    dataloader_train = DataLoader(train_data, batch_size = args.gpu_train_batch_size,  sampler=train_sampler)

    ## review data 배분
    if args.review:

        review_sampler = DistributedSampler(review_data, num_replicas=world_size, rank=rank, shuffle=True, seed=args.twiki_num) # 매번 다르게 샘플링되도록
        dataloader_review = DataLoader(review_data, batch_size = int(args.gpu_train_batch_size * args.review_ratio),  sampler=review_sampler)

    else:
        review_data = train_data # 임시지정
        dataloader_review = dataloader_train


        
    ## eval_data_changed 배분  
    if eval_data_changed.num_rows  % world_size ==0:
        block_size = (eval_data_changed.num_rows // world_size) #
    else:
        block_size = (eval_data_changed.num_rows // world_size) + 1
    eval_data_changed = eval_data_changed.select(range(block_size*rank , min(block_size*(rank+1), eval_data_changed.num_rows)))
    dataloader_changed= DataLoader(eval_data_changed,shuffle=False ,batch_size=args.eval_batch_size)

    ## eval_data_unchanged 배분  
    if eval_data_unchanged.num_rows  % world_size ==0:
        block_size = (eval_data_unchanged.num_rows // world_size) #
    else:
        block_size = (eval_data_unchanged.num_rows // world_size) + 1
    eval_data_unchanged = eval_data_unchanged.select(range(block_size*rank , min(block_size*(rank+1), eval_data_unchanged.num_rows)))
    dataloader_unchanged= DataLoader(eval_data_unchanged,shuffle=False ,batch_size=args.eval_batch_size)

    ##################
    if args.meta_type=='lamackl':
        print(f"{rank} train: {train_data.num_rows // (world_size * args.gpu_train_batch_size )} step / {train_data.num_rows // (world_size * args.gpu_train_batch_size * args.train_grad_accum_step )} update/ review: {review_data.num_rows} / {eval_data_unchanged.num_rows}")

    elif args.meta_type=='twiki':
        print(f"{rank} train: {train_data.num_rows // (world_size * args.gpu_train_batch_size )} step / {train_data.num_rows // (world_size * args.gpu_train_batch_size * args.train_grad_accum_step )} update/ review: {review_data.num_rows // world_size } / {eval_data_changed.num_rows} / {eval_data_unchanged.num_rows}")

    # ## initalize
    # model = Model_wTA(args).to(rank)  # rho때문에 위로 올림


    ddp_model = DDP(model, device_ids=[rank])
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # if args.rho:
        # model.theta.eval()
        # with torch.no_grad():



    if args.recadam:
        recadam_anneal_w = 1.0
        recadam_anneal_fun = 'sigmoid'
        recadam_anneal_k = 0.5
        recadam_anneal_t0 = 250
        recadam_pretrain_cof = 5000.0
        no_decay={'params':[], 'pretrain_params':[], 'weight_decay':0.01, 'anneal_w': recadam_anneal_w}
        with_decay={'params':[], 'pretrain_params':[], 'weight_decay':0.0, 'anneal_w':0.0}
        for name, param in model.named_parameters():
            if param.requires_grad:
                if any(nd in name for nd in ['bias','norm']):
                    no_decay['params'].append(param)
                    no_decay['pretrain_params'].append(param.clone().detach())
                else:
                    with_decay['params'].append(param)
                    with_decay['pretrain_params'].append(param.clone().detach())
        optimizer = RecAdam([no_decay, with_decay], lr=args.train_lr ,
                                anneal_fun=recadam_anneal_fun, anneal_k=recadam_anneal_k,
                                anneal_t0=recadam_anneal_t0, pretrain_cof=recadam_pretrain_cof)

    else:
        param_list = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                param_list.append(param)
        optimizer = AdamW(param_list, lr=args.train_lr, is_paged=True)

    # rand_ids = random.Random()
    # review_rand_ids = random.Random()
    # rand_ids.seed(1)
    # review_rand_ids.seed(1)
    tqdm_disable = False if rank==0 else True

    # if rank==0:
        # for name, param in model.named_parameters():
            # print(f"{param.dtype} {param.requires_grad} {name}")

    
    loss_mean=torch.tensor(0.).to('cuda') #  epoch=-1 에서 eval 하기위함
    for epoch in range(-1, args.max_epochs):  
        if args.test_type=='targeted':
            model.phi.eval()
        model.theta.train()
        #     optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        if epoch >-1:

            optimizer.zero_grad()
            loss_list=[]
            for step, (batch_data, review_batch_data) in tqdm(enumerate(zip(dataloader_train, dataloader_review)), disable=tqdm_disable):
                
                if args.meta_type=='twiki':
                    key0 = 'text'
                elif args.meta_type=='lamackl':
                    key0 = 'evidence'
                evidences = batch_data[key0]
                if args.review:
                    # evidences = evidences + review_batch_data['evidence']
                    evidences = evidences + review_batch_data[key0]

                tokens = tokenizer(evidences, return_tensors='pt', truncation=True, padding=True, max_length=args.token_max_length).to('cuda')
                
                if args.rho:
                    loss = ddp_model(tokens, batch_data=batch_data, mode0='rho_train')
                elif args.rho_model:
                    loss = ddp_model(tokens, batch_data=batch_data, mode0='rho_model_train')
                elif args.test_type=='oracle':
                    object_list= batch_data['object']
                    object_target_tokens =  tokenizer(object_list, add_special_tokens=False).input_ids

                    label_masks = giveme_label_mask_list(tokens, object_target_tokens).to('cuda')                 

                    loss = ddp_model(tokens, mode0='train', weight_vector=label_masks)                       
                else:
                    loss = ddp_model(tokens , mode0='train')

                if torch.isnan(loss).any():
                    print("nan 있음")
                loss = loss / args.train_grad_accum_step # default 1

                loss.backward()
                loss_list.append(loss.clone().detach())

                # gradiant accumulation # accum_step=1  이면 accum 안함
                if ((step + 1) % args.train_grad_accum_step ==0) or (step + 1) == len(dataloader_train) :
                    # if rank==0:
                        # print(f"{step+1}  {len(dataloader_train)}")
                    optimizer.step()
                    optimizer.zero_grad()

            with torch.no_grad():
                loss_count = torch.tensor(len(loss_list), device=rank)
                loss_mean = torch.stack(loss_list).detach().sum()

                dist.all_reduce(loss_mean, op=dist.ReduceOp.SUM)
                dist.all_reduce(loss_count, op=dist.ReduceOp.SUM)
                loss_mean = loss_mean/loss_count
            if rank==0:
                print(f"{epoch} epoch / Loss : {loss_mean}") ## loss.item() : loss 값
                

            dist.barrier()



        ######eval#####
        model.theta.eval()
        # torch.cuda.empty_cache()
        if rank==0:
            print("eval 시작")
        with torch.no_grad():
            ###### 1. changed#####
            total_ppl = torch.tensor([0.0], device=rank)
            total_acc = torch.tensor([0.0], device=rank)
            total_count = torch.tensor([0], device=rank)
            for ev_step, batch in enumerate(dataloader_changed):
                # print(f"================{ev_step}===========")
                object_list= batch['object']
                object_target_tokens =  tokenizer(object_list, add_special_tokens=False).input_ids

                task_data = make_task(args, batch)
                task_token = tokenizer(task_data, return_tensors='pt', truncation=True, padding=True).to('cuda')
                label_masks = giveme_label_mask_list(task_token, object_target_tokens).to('cuda')

                target_ppl, target_acc = ddp_model(task_token , mode0='eval', data={'label_masks':label_masks, 'object_target_tokens':object_target_tokens})
                total_ppl += target_ppl.sum().to(rank)
                total_acc += target_acc.sum().to(rank)
                total_count += target_acc.numel()
            dist.all_reduce(total_ppl, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_acc, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_count, op=dist.ReduceOp.SUM)
            mean_ppl_changed = total_ppl / total_count
            mean_acc_changed = total_acc / total_count

            dist.barrier()
                

            ## 2. unchanged
            total_ppl = torch.tensor([0.0], device=rank)
            total_acc = torch.tensor([0.0], device=rank)
            total_count = torch.tensor([0], device=rank)

            for batch in dataloader_unchanged:
                object_list= batch['object']
                object_target_tokens =  tokenizer(object_list, add_special_tokens=False).input_ids

                task_data = make_task(args, batch)
                task_token = tokenizer(task_data, return_tensors='pt', truncation=True, padding=True, max_length=512).to('cuda')
                label_masks = giveme_label_mask_list(task_token, object_target_tokens).to('cuda')

                target_ppl, target_acc = ddp_model(task_token , mode0='eval', data={'label_masks':label_masks, 'object_target_tokens':object_target_tokens})

                total_ppl += target_ppl.sum().to(rank)
                total_acc += target_acc.sum().to(rank)
                total_count += target_acc.numel()
            dist.all_reduce(total_ppl, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_acc, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_count, op=dist.ReduceOp.SUM)
            mean_ppl_unchanged = total_ppl / total_count
            mean_acc_unchanged = total_acc / total_count
            
            if rank==0:
                # self.model
                init_names= []
                init_params = []
                for name, param in model.theta.named_parameters():
                    if param.requires_grad:
                        init_names.append(name)
                        init_params.append(param.clone().detach().cpu())

                eval_result[epoch] = {'ppl_changed': mean_ppl_changed[0].cpu(), 'acc_changed': mean_acc_changed[0].cpu(), 'ppl_unchanged':mean_ppl_unchanged[0].cpu(), 'acc_unchanged':mean_acc_unchanged[0].cpu(), 'loss':loss_mean.detach().cpu(),}# 'checkpoint':{'names':init_names, 'params':init_params}}

                print(f"{epoch}, ch: ppl {mean_ppl_changed[0].cpu() :.6f}, acc {mean_acc_changed[0].cpu() :.6f} / unch : ppl {mean_ppl_unchanged[0].cpu() :.6f} , acc {mean_acc_unchanged[0].cpu() :.6f}")

                ## save
                if args.save_on_eval:
                    result_dic = {'config': {** args.__dict__},
                    'eval_result':eval_result}
                    result_dic['config']['world_size'] = world_size

                    with open(f'results/{args.meta_type}/{args.test_type}_{args.model_type}{"_review" if args.review else ""}{"_recadam" if args.recadam else ""}{"_"+ args.add_to_title if args.add_to_title != "" else ""}.pkl', 'wb') as f:
                        pickle.dump(result_dic, f)

                

        ####  마지막에도 저장하도록 했음.  
        if args.meta_type == 'twiki' and args.twiki_temp_save  and epoch > -1 :
        # if args.meta_type == 'twiki' and args.twiki_temp_save and (args.twiki_num < args.twiki_num_last) and epoch > -1 :
            if rank==0:
                with torch.no_grad():
                    theta_names = []
                    theta_params = []
                    for name, param in model.theta.named_parameters():
                        if param.requires_grad:
                            theta_names.append(name)
                            theta_params.append(param.detach().to('cpu'))
                save_dic = {'theta_params':theta_params, 'theta_names':theta_names}
                print("temp 저장")
                with open(f'./temp/temp_{args.test_type}_{args.model_type}{"_review" if args.review else ""}{"_recadam" if args.recadam else ""}{"_"+ args.add_to_title if args.add_to_title != "" else ""}{"_"+ str(args.twiki_num)}.pkl', 'wb') as f:
                    pickle.dump(save_dic, f)

        dist.barrier()
        
    cleanup()


def run_demo(demo_fn, world_size, train_data, eval_data_changed, eval_data_unchanged,  review_data, unchanged_ppl, eval_result):
    mp.spawn(demo_fn,  # demo_fn  이  5번파일 run() 과 같음
            args=(world_size, train_data, eval_data_changed, eval_data_unchanged,  review_data, unchanged_ppl, eval_result),
            nprocs=world_size,
            join=True)
    
 

if __name__ == "__main__":
    n_gpus = torch.cuda.device_count() # 타이탄 서버는 3개
    # assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus

    # def shorten(example):
        # example['text'] = example['text'][:1000]
        # return example
    train_data = load_dataset(path= os.path.dirname(args.train_data),data_files=[os.path.basename(args.train_data)])['train']#.select(range(100))
    # train_data = train_data.map(shorten)


    eval_data_unchanged = load_dataset(path= os.path.dirname(args.eval_data_unchanged),data_files=[os.path.basename(args.eval_data_unchanged)])['train']#.select(range(10))

    eval_data_changed = load_dataset(path= os.path.dirname(args.eval_data_changed),data_files=[os.path.basename(args.eval_data_changed)])['train']#.select(range(10))




    if args.review==True:
        if args.review_data != 'none':
            review_data = load_dataset(path= os.path.dirname(args.review_data))['train']#.select(range(10))
    else:
        review_data = train_data


    with mp.Manager() as manager:
        # learned_result = manager.list()
        learned_result = manager.dict()
        no_forget_result = manager.dict()
        loss_result = manager.dict()

        unchanged_ppl = manager.list()
        eval_result= manager.dict()

        run_demo(demo_basic, world_size, train_data, eval_data_changed, eval_data_unchanged,  review_data, unchanged_ppl, eval_result)

        eval_result = dict(eval_result)
        result_dic = {'config': {** args.__dict__},
        'eval_result':eval_result}
        result_dic['config']['world_size'] = world_size

        with open(f'/results/{args.meta_type}/{args.test_type}_{args.model_type}{"_review" if args.review else ""}{"_recadam" if args.recadam else ""}{"_"+ args.add_to_title if args.add_to_title != "" else ""}{"_"+ str(args.twiki_num) if args.twiki_temp_save else ""}.pkl', 'wb') as f:
            pickle.dump(result_dic, f)
