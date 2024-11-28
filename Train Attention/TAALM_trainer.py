# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import os
from transformers import AutoTokenizer
import torch
import torch.nn as nn
import copy
from tqdm import tqdm
from utils.tool_TAALM.taalm_adamw import AdamW as taalm_AdamW
from TAALM import TAALM


def TAALM_train(args,
                phi,
                gamma,
                theta,
                phi_calcul_params,
                theta_init_names,
                theta_init_params,
                Data, # data
                TD, # task on data
                TU, # unrelated task
                dummy_loop,
                inner_max_step,
                w_regul=False,
                known_masking=False,
                TD_label = False,
                TD_label_mask = None
               ):
    
    ## outer train
    ## theta
    theta_calcul_params = [param.clone().detach().requires_grad_() for param in theta_init_params]
    theta_optimizer = taalm_AdamW(theta_calcul_params, lr=2e-4)

    TAALM.inject_param(theta, theta_calcul_params, theta_init_names, 'assign')

    with torch.no_grad():
    ### zero ppl(TD)
        # after_gamma = gamma(TD.input_ids)
        temp_inputs = TD.copy()
        temp_inputs['labels'] = temp_inputs['input_ids'] # origin_loss 계산 위한것
        # temp_inputs['input_ids'] = after_gamma
        outputs = theta(**temp_inputs)
        log_probs = -nn.functional.log_softmax(outputs.logits[:,:-1,:], dim=-1)
        labels0 = torch.clamp(TD['input_ids'][:,1:], min=0).unsqueeze(-1)
        nll_loss = log_probs.gather(dim=-1, index=labels0).squeeze(-1)
        zero_ppl_TD_each = nll_loss * TD['attention_mask'][:,:-1]
        # zero_ppl_D= torch.div(torch.sum(zero_ppl_D_each, dim=-1) , torch.sum(tt_task['attention_mask'][:,:-1], dim=-1)).sum()

        if args.do_restrict:
            ## unrelated  task
            # after_gamma = gamma(TU.input_ids)
            temp_inputs = TU.copy()
            temp_inputs['labels'] = temp_inputs['input_ids'] # origin_loss 계산 위한것
            # temp_inputs['input_ids'] = after_gamma
            outputs = theta(**temp_inputs)
            log_probs = -nn.functional.log_softmax(outputs.logits[:,:-1,:], dim=-1)
            labels0 = torch.clamp(TU['input_ids'][:,1:], min=0).unsqueeze(-1)
            nll_loss = log_probs.gather(dim=-1, index=labels0).squeeze(-1)
            zero_unrelated = nll_loss * TU['attention_mask'][:,:-1]    

    #### phi 가 target_weight 를 생산하게 함 ####
    # after_gamma = gamma(Data.input_ids)
    temp_inputs = Data.copy()
    # temp_inputs['input_ids'] = after_gamma
    output= phi(**temp_inputs)
    logit0 = output.logits.squeeze(-1)[:,1:]
    w_action = torch.sigmoid(logit0)

    if w_regul:
        
        w_action = w_action / torch.max(w_action, dim=-1).values.view(-1,1)


    for inner_epoch in range(dummy_loop + inner_max_step):
        not_dummy=False if inner_epoch < dummy_loop else True

        theta_loss, loss_origin, w_loss_sepa, _ = TAALM.calcul_loss(gamma = gamma, theta=theta, inputs=Data, tt_weight_vector=w_action, tt_target_weight=True, tt_L2=False, known_masking=known_masking, onepiece=True)

        grad0 = torch.autograd.grad(theta_loss, theta_calcul_params, create_graph=not_dummy) # dummy 일때는 False,  아닐떄는 True
        
        ## theta_calcul_params update , inject into theta
        assert(len(theta_calcul_params) == len(grad0))
        theta_calcul_params = theta_optimizer.step(grad0, theta_calcul_params)

        # init이 아닌 calcul_params임
        if not_dummy:
            TAALM.inject_param(theta, theta_calcul_params, theta_init_names, 'assign')

        else: ## dummy
            TAALM.inject_param(theta, theta_calcul_params, theta_init_names, 'detach_reqgrad')

        
        if not_dummy: # 
            pass
        else: # dummy
            theta_calcul_params=[]
            TAALM.extract_param(theta, theta_calcul_params, theta_init_names, 'assign')

        # print(theta_loss.detach(), end=' | ')
    ## Let theta' solve TD 
    ## outer train
    if TD_label:
        phi_loss, loss_origin, w_loss_sepa, nll_loss = TAALM.calcul_loss(gamma=gamma, theta=theta, inputs=TD, tt_weight_vector=TD_label_mask, tt_target_weight=True, tt_L2=False, TD_label=False, onepiece=True) 
        
    else:
        phi_loss, loss_origin, w_loss_sepa, nll_loss = TAALM.calcul_loss(gamma=gamma, theta=theta, inputs=TD, tt_weight_vector=torch.tensor(1), tt_target_weight=False, tt_L2=False, onepiece=True)

    # change ^3
    change_pow = torch.pow(nll_loss - zero_ppl_TD_each, 3)
    if TD_label:
        change_pow = change_pow * TD_label_mask
        change_loss = torch.div(torch.sum(change_pow, dim=-1) , torch.sum(TD_label_mask, dim=-1)).sum()
    change_loss = torch.div(torch.sum(change_pow, dim=-1) , torch.sum(TD['attention_mask'], dim=-1)).sum()

    
    if args.obj_type=='phi_loss':
        obj_temp = phi_loss
    elif args.obj_type=='change3':
        obj_temp = change_loss 
    elif args.obj_type=='change2':
        change0 = nll_loss - zero_ppl_TD_each
        change_pow = torch.pow(change0, 2) * torch.sign(change0)
        change_loss = torch.div(torch.sum(change_pow, dim=-1) , torch.sum(TD['attention_mask'], dim=-1)).mean()
        obj_temp=change_loss


    if args.do_restrict:
        ## restrict unrelated
        _, loss_origin, w_loss_sepa, nll_loss = TAALM.calcul_loss(gamma=gamma, theta=theta, inputs=TU, tt_weight_vector=torch.tensor(1), tt_target_weight=False, tt_L2=False, onepiece=True)
        unrelate_restrict = torch.abs(nll_loss - zero_unrelated)
        unrelate_restrict = torch.div(torch.sum(unrelate_restrict, dim=-1) , torch.sum(TU['attention_mask'], dim=-1)).mean()
        
        ##
        
        objective_function = obj_temp + unrelate_restrict
    else:
        unrelate_restrict= torch.tensor(0.)
        objective_function = obj_temp

    grad1 = torch.autograd.grad(objective_function, phi_calcul_params)
    torch.cuda.empty_cache() # 
    
    return grad1, {'phi_loss':phi_loss.detach(), 'change_loss':change_loss.detach(), 'unrelate_restrict':unrelate_restrict.detach(), 'objective_function':objective_function.detach(),
                   'weight_mu':w_action.detach(),
                   'Data':Data.input_ids}