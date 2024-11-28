import torch
from ta_toolbox import giveme_label_mask
from TAALM import TAALM
from utils.tool_TAALM.taalm_adamw import AdamW as taalm_AdamW
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

def default_for_score() : return{'phi_loss':[], 'change_loss':[], 'stop_word_loss':[],'propn_ppl':[], 'propn_acc':[]}



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
def giveme_label_mask0(query_token, label_token):  
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
        target_mask= giveme_label_mask0(task, torch.tensor(target).to('cuda'))
        target_masks.append(target_mask[1:])

    target_masks = torch.stack(target_masks)
    return target_masks
#####
def eval_forward(theta, token, data):
    with torch.no_grad():
        response=theta(**token)
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

    return task_targeted_loss, torch.stack(scores)

def trex_eval_batch(args,  # batch 전체 학습
            phi,
            gamma,
            theta,
            theta_init_names,
            theta_init_params,
            dataset,
            tokenizer):
    
    h_temp_val = {'val_weight_mu':[], 'val_Data':[]}
    history_for_eval = {'theta_loss_each':[]}
    result_ppl = []
    result_acc = []

    eval_gen = torch.Generator()
    eval_gen.manual_seed(42)
    dataloader_eval = DataLoader(dataset, shuffle=True, batch_size=args.eval_batch_size, generator=eval_gen)

    TAALM.inject_param(theta, theta_init_params, theta_init_names, 'detach_reqgrad')
    eval_optimizer = torch.optim.AdamW(theta.parameters(), lr=2e-4)


    ##### train theta #####
    theta.train()
    phi.eval() 
    first0 = True
    for inner_epoch in tqdm(range(args.val_dummy_loop + 1)):
        for step, batch_data in enumerate(dataloader_eval):
            if args.meta_type=='twiki':
                evidences = batch_data['text']
            elif args.meta_type=='lamackl':
                evidences = batch_data['evidence']

            subject_list = batch_data['subject']

            Data = evidences

            Data = tokenizer(Data, return_tensors='pt', truncation=True, padding=True, max_length=args.token_max_length).to('cuda')
            
            ## predict multiple train-attention
            with torch.no_grad():
                output = phi(**Data)
                logit0 = output.logits.squeeze(-1)[:,1:]
                w_action = torch.sigmoid(logit0) * Data.attention_mask[:,:-1]

            loss, loss_origin, w_loss_sepa, _ = TAALM.calcul_loss(gamma=None, theta=theta, inputs=Data, tt_weight_vector=w_action, tt_target_weight=True, tt_L2=False,  known_masking=True, onepiece=True)
            loss = loss / args.eval_grad_accum_step # default 1
            loss.backward()
            if ((step + 1) % args.eval_grad_accum_step ==0) or (step + 1) == len(dataloader_eval) :
                eval_optimizer.step()
                eval_optimizer.zero_grad()

            with torch.no_grad():
                if first0:
                    ### 저장용
                    h_temp_val['val_Data'].extend(Data.input_ids.detach().cpu())
                    h_temp_val['val_weight_mu'].extend(w_action.detach().cpu())
        first0=False
    # h_temp_val['val_Data'] = torch.stack(h_temp_val['val_Data']) # 길이가 안맞아서 stack이 안됨
    # h_temp_val['val_weight_mu'] = torch.stack(h_temp_val['val_weight_mu'])
    ###### Evaluation with target-trained theta ####
    theta.eval()
    with torch.no_grad():
        total_ppl = torch.tensor([0.0]).to('cuda')
        total_acc = torch.tensor([0.0]).to('cuda')
        total_count = torch.tensor([0]).to('cuda')
        dataloader_eval = DataLoader(dataset, shuffle=False, batch_size=args.eval_batch_size)

        for batch in dataloader_eval:
            task_data = make_task(args, batch)
            object_list= batch['object']
            # task_data = batch[f'task_{args.task_type}']
            object_target_tokens =  tokenizer(object_list, add_special_tokens=False).input_ids
            task_token = tokenizer(task_data, return_tensors='pt', truncation=True, padding=True).to('cuda')
            label_masks = giveme_label_mask_list(task_token, object_target_tokens).to('cuda')

            target_ppl, target_acc = eval_forward(theta, task_token , data={'label_masks':label_masks, 'object_target_tokens':object_target_tokens})
            total_ppl += target_ppl.sum()
            total_acc += target_acc.sum()
            total_count += target_acc.numel()

        mean_target_ppl = total_ppl / total_count
        mean_target_acc = total_acc / total_count

    return mean_target_ppl[0].cpu(),  mean_target_acc[0].cpu(), h_temp_val

