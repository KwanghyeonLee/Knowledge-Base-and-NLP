
import os
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'  # 토크나이저 워닝 지움
from datasets import load_dataset, Dataset
import random
import torch
from TAALM_trainer import TAALM_train
from TAALM import TAALM
from utils.tool_TAALM.taalm_adamw import AdamW as taalm_AdamW
from transformers import AutoTokenizer   
from ta_toolbox import make_conv, giveme_label_mask
# from old_versions.tt_MSC_eval import Multi_Session_Ppl
from collections import OrderedDict
import pickle 
from datetime import datetime
from ta_eval import trex_eval_batch

import wandb
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--meta_type', type=str, default='none') # 
parser.add_argument('--val_size', type=int)
parser.add_argument('--eval_batch_size', type=int)
parser.add_argument('--eval_grad_accum_step', type=int, default=1)
parser.add_argument('--val_interval',type=int)
parser.add_argument('--batch_size',type=int)
parser.add_argument('--train_size',type=int)
parser.add_argument('--train_data',type=str)
parser.add_argument('--do_restrict',type=str, default='false')
parser.add_argument('--phi_regul',type=str, default='false')
parser.add_argument('--phi_model_name',type=str, default='false')
parser.add_argument('--phi_adapter_file',type=str, default='false')
parser.add_argument('--theta_model_name',type=str, default='false')
parser.add_argument('--theta_adapter_file',type=str, default="qlora")
# parser.add_argument('--phi_loss_2_obj',type=str, default='false')
parser.add_argument('--obj_type',type=str, default='change3')
parser.add_argument('--dummy_loop',type=int, default=14)
parser.add_argument('--val_dummy_loop',type=int, default=14)
parser.add_argument('--history_path')
parser.add_argument('--known_masking', type=bool, default=False)
parser.add_argument('--val_known_masking', type=bool, default=False)
parser.add_argument('--task_type', type=str, default='none') # descriptive, schematic
parser.add_argument('--TD_label', type=bool, default=False) # phi 를 task label을 씌워서 훈련. 
parser.add_argument('--token_max_length', type=int, default=-1) # phi 를 task label을 씌워서 훈련. 

args = parser.parse_args()

outer_max_epochs = 30  # 원래의 outer_max_step 과 다름. dataset 전체를 몇바퀴 돌것이냐.
batch_size = args.batch_size
dummy_loop = args.dummy_loop #14
inner_max_step = 1
eval_inner_max_step = args.val_dummy_loop + inner_max_step
alpha_lr = 2e-4
beta_lr = 2e-4
valid_interval = args.val_interval
phi_regul = args.phi_regul
valid_know_masking = False
valid_w_regul = True
theta_adapter_file = args.theta_adapter_file


do_restrict = True if args.do_restrict.lower()=='true' else False
args.do_restrict = do_restrict
phi_regul = True if args.phi_regul.lower()=='true' else False
args.phi_regul = phi_regul



# do_restrict = False

config0 = {'batch_size':batch_size,
'dummy_loop':dummy_loop,
'inner_max_step':inner_max_step,
'eval_inner_max_step':eval_inner_max_step,
'alpha_lr':alpha_lr,
'beta_lr':beta_lr,
'valid_interval':valid_interval,
'do_restrict': do_restrict,
'phi_regul':phi_regul,
 **args.__dict__}
print(config0)
print(args)
run = wandb.init(
    project='trex_MAML',
    # name='weighted_celoss',
    name=f'20240413_{args.obj_type}',
    config=config0,
    notes="twiki  phi 학습 ",
    tags=['continual learning'],
)


#gen_model  (gamma + theta)
theta = TAALM.init_theta(adapter_file=theta_adapter_file, model_name=args.theta_model_name, onepiece=True)
#tw_model (gamma + phi)
phi = TAALM.init_phi(regul=phi_regul, model_name=args.phi_model_name, adapter_file=args.phi_adapter_file, onepiece=True)

torch.cuda.empty_cache()
phi = TAALM.reboot_phi_model(phi, regul=phi_regul, adapter_file=args.phi_adapter_file)
theta = TAALM.reboot_theta_model(theta, adapter_file=args.theta_adapter_file)

gamma = None



### score board
score_records = OrderedDict()
    # valid 안에서  score_board -> score_pallet -> scores_result 순서로 취합돼 score_records 안으로 들어옴


########## Param and Optimizer ############
#========= theta (generator)
theta_init_params=[]
theta_init_names=[]
TAALM.extract_param(theta, theta_init_params, theta_init_names, 'clone_detach')
TAALM.inject_param(theta, theta_init_params, theta_init_names, 'clone')

#========= phi (train-attention)
phi = TAALM.reboot_phi_model(phi, regul=phi_regul, adapter_file=args.phi_adapter_file)
phi_calcul_params = [] 
phi_init_names = []
TAALM.extract_param(phi, phi_calcul_params, phi_init_names, 'assign')
phi_optimizer = taalm_AdamW(phi_calcul_params, lr=beta_lr)
#####################################################
        
# tokenizer
model_name = "meta-llama/Llama-2-7b-hf"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token  

        
def preprocess(example):
    if len(example['sessions']) >= 3:
        return example
def make_real_id(example, index0):
    example['real_id'] = index0
    return example



############## dataset ##################

dataset = load_dataset('json', data_files=f'./data/{args.train_data}.jsonl', split='train') ## 
dataset = dataset.train_test_split(test_size=args.val_size, seed=42)  # 1013 : 100     train : test

#########################################################

rand_ids = random.Random()  # id 순서
rand_ses = random.Random()  # session 아이디
rand_unr = random.Random()  # restrict 아이디
rand_unr_ses = random.Random()  # restrict의 세션 id
rand_ids.seed(1)
rand_ses.seed(1)
rand_unr.seed(1)
rand_unr_ses.seed(1)
step_n = 0

history = {'Data':[],'weight_mu':[], 'theta_loss':[], 'phi_loss':[], 'change_loss':[], 'objective_function':[], 'unrelate_restrict':[], 'val_Data':[], 'val_weight_mu':[],'lm_head_bias':[], 'val_score':[], 'val_ppl':[]}
first0=True
for epoch in range(outer_max_epochs):
    ### epoch
    train_dataset_ids = list(range(dataset['train'].__len__()))
    rand_ids.shuffle(train_dataset_ids)
    for start_id0 in range(0, len(train_dataset_ids), batch_size):

        ### phi refresh ===================================
        phi_calcul_params = [param.detach().requires_grad_() for param in phi_calcul_params]
        TAALM.inject_param(phi, phi_calcul_params, phi_init_names, 'assign')

        ####===============================================

        ### step
        grad_list = [torch.tensor(0.).to('cuda')] * len(phi_calcul_params)
        grad_n = 0

        h_temp = {'Data':[],'weight_mu':[], 'theta_loss':[], 'phi_loss':[], 'change_loss':[], 'objective_function':[], 'unrelate_restrict':[]}
        train_batch_ids = train_dataset_ids[start_id0: start_id0 + batch_size]
        
        if not first0:
            for id0 in train_batch_ids:
                subject = dataset['train'][id0]['subject']
                object = dataset['train'][id0]['object']
                
                if args.meta_type =='lamackl':
                    Data = dataset['train'][id0]['evidence']
                    TD = dataset['train'][id0][f"task_{args.task_type}"]
                elif args.meta_type == 'twiki':
                    relation = dataset['train'][id0]['relation']

                    Data = dataset['train'][id0]['text']
                    TD = f"Guess the object. \n  subject is {subject} , relation is {relation} , object is {object}"

                Data = tokenizer([Data], return_tensors='pt',truncation=True, padding=True, max_length=args.token_max_length).to('cuda')
                TD = tokenizer([TD], return_tensors='pt',truncation=True, padding=True).to('cuda')

                ###### label_mask for TD ####
                label = torch.tensor(tokenizer(object).input_ids[1:]).to('cuda')
                TD_label_mask = giveme_label_mask(TD, label).to('cuda')[1:]
                ############################



                ## grad 계산
                grad1, logs0 = TAALM_train(
                    args = args,
                    phi=phi,
                    gamma=gamma,
                    theta=theta,
                    phi_calcul_params = phi_calcul_params,
                    theta_init_names = theta_init_names,
                    theta_init_params = theta_init_params,
                    Data =Data,
                    TD = TD,
                    TU = None,
                    dummy_loop = dummy_loop,
                    inner_max_step = inner_max_step,
                    w_regul = False,
                    known_masking = False,
                    TD_label= args.TD_label,
                    TD_label_mask=TD_label_mask
                )

                ## accumulate grad in batch
                # grad_list.append(grad1)
                with torch.no_grad():
                    grad_list = [p_a + p_b for p_a, p_b in zip(grad_list, grad1)]
                    grad_n +=1

                ## prepare to write history==================
                with torch.no_grad():
                    for k0, v0 in logs0.items():
                        
                        h_temp[k0].append(v0.cpu())
                ##===============================================

            with torch.no_grad():
                grad_update = [x/grad_n for x in grad_list]

            ## phi optimizer step with  grad average
            assert(len(phi_calcul_params) == len(grad_update))
            phi_calcul_params = phi_optimizer.step(grad_update, phi_calcul_params)
            
            # inject the params again into the phi
            TAALM.inject_param(phi, phi_calcul_params, phi_init_names, 'detach_reqgrad')
            phi_calcul_params = []
            TAALM.extract_param(phi, phi_calcul_params, phi_init_names, 'assign')
            torch.cuda.empty_cache()

            
            ## write history==============================
            with torch.no_grad():
                for k0, v0 in h_temp.items():
                    history[k0].append(v0)
                history['lm_head_bias'] = phi.lm_head.bias
            ## ============================================        
            step_n +=1
            with torch.no_grad():
                print(f"{epoch}, {step_n },  {torch.stack(h_temp['phi_loss']).mean() :.7}  , {torch.stack(h_temp['change_loss']).mean() :.7} , {torch.stack(h_temp['unrelate_restrict']).mean() :.7} , {torch.stack(h_temp['objective_function']).mean() :.7}, {phi.lm_head.bias}")
            
            with open(f"{args.history_path}/history.pkl", 'wb') as f:
                pickle.dump(history, f)
        first0= False
        
        #### validation
        if step_n % valid_interval == 0:

            phi.eval()
            print(f"step {step_n}, start validation.")
            #### validation()

            start0 = datetime.now()
            
            ppl, score, h_temp_val = trex_eval_batch(args, phi, gamma, theta, theta_init_names, theta_init_params, dataset['test'], tokenizer)
            phi.train()
            theta.train()
                
            with torch.no_grad():
                phi_to_save = [param.detach().to('cpu') for param in  phi_calcul_params]

            save_dic = {'val_score':score, 'val_ppl':ppl,  'phi_params':phi_to_save, 'phi_names':phi_init_names}
            with open(f"{args.history_path}/checkpoint_{step_n}.pkl", 'wb') as f:
                pickle.dump(save_dic, f)
            
            end0 = datetime.now()

            ## write history==============================
            with torch.no_grad():
                history['val_ppl'].append(ppl)
                history['val_score'].append(score)

                for key0 in h_temp_val:
                    history[key0].append(h_temp_val[key0]) # Data, weight 추가

            with open(f"{args.history_path}/history.pkl", 'wb') as f:
                pickle.dump(history, f)
            ## ============================================  
            print(f"{step_n}, score: {score} , ppl: {ppl}")
            print(f"Now:{end0.strftime('%Y-%m-%d %H:%M:%S')} :  took  {end0-start0} for validation")