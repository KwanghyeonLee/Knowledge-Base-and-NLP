import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, LlamaConfig, AutoConfig
from torch import nn
from torch import Tensor as T
from typing import Tuple, List
import transformers
import copy
from collections import OrderedDict
from trl import SFTTrainer
from transformers import TrainingArguments
from peft import LoraConfig, get_peft_model 
from utils.kadapter.Llama2_Model_Kadapter import Llama2LMHeadModel
import bitsandbytes as bnb

import logging 



def find_all_linear_names( model):
    a = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, a):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])


    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


#### 
class TAALM(AutoModelForCausalLM):
    def __init__(self):
        AutoModelForCausalLM.__init__(self)

    @classmethod
    def init_theta(cls, adapter_file='qlora', onepiece=False, model_name="meta-llama/Llama-2-7b-hf", lm_train=False, qlora_lora='qlora', **kwargs)-> AutoModelForCausalLM:
        torch.manual_seed(42)

        bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",  
        bnb_4bit_compute_dtype=torch.bfloat16,  
        )

        if 'llama' in model_name.lower():
            config = LlamaConfig(model_name =model_name,
                             _attn_implementation = 'eager',)
        else:
            config = AutoConfig.from_pretrained(model_name)

        if model_name=="TinyLlama/TinyLlama-1.1B-Chat-v1.0":
            config.hidden_size = 2048
            config.intermediate_size = 5632
            config.num_hidden_layers=22
            config.num_key_value_heads = 4

        if qlora_lora=="lora":
            bnb_config = None

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config = bnb_config,
            trust_remote_code = True,
            config=config,
            )
        model.config.use_cache = False
        lora_alpha = 16
        lora_dropout = 0.1
        lora_r = 64   #

        modules = find_all_linear_names(model)
        if qlora_lora in ['small_qlora', 'lora']:
            modules = None

        peft_config = LoraConfig(
            lora_alpha = lora_alpha,
            lora_dropout= lora_dropout,
            r=lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=modules, 
        )

        model = get_peft_model(model, peft_config)

        if qlora_lora=='qlora':
            model.delete_adapter('default')
            model.model.load_adapter(f'Anonymous-TAALM/{adapter_file}', adapter_name='default') 
            model.model.set_adapter('default')
            

        for name, module in model.named_modules():
            if "norm" in name :#or 'embed' in name:
                module = module.to(torch.float32)

        if 'gpt' in model_name:
            for name, param in model.named_parameters():
                if 'ln' in name:
                    param.data = param.data.to(torch.float32)

        for name, param in model.named_parameters():
            if 'lora' in name :
                param.requires_grad = True
            else:
                param.requires_grad = False

        if lm_train:
            for name, param in model.named_parameters():
                if '.norm' in name or 'lm_head' in name:
                    param.requires_grad = True

        

        if 'llama' in model_name.lower():
            model.model.model.embed_tokens.to(torch.float32)

        if 'llama' in model_name.lower():
            model.model.model.embed_tokens.requires_grad=False
        model.model.lm_head.to(torch.float32)

        return model
        

    @classmethod
    def init_phi(cls, adapter_file='qlora', regul=False, onepiece=False,model_name="meta-llama/Llama-2-7b-hf", **kwargs)-> AutoModelForCausalLM:
        torch.manual_seed(42)
        bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",   # data type
        bnb_4bit_compute_dtype=torch.float32,  # compute data type
        )

        if 'llama' in model_name.lower():
            config = LlamaConfig(model_name = model_name,
                             _attn_implementation = 'eager')#### _attn_implementation = 'eager'  이거 해야 2차미분 됨 #####
        else:
            config = AutoConfig.from_pretrained(model_name)

            
        if model_name=="TinyLlama/TinyLlama-1.1B-Chat-v1.0":
            config.hidden_size = 2048
            config.intermediate_size = 5632
            config.num_hidden_layers=22
            config.num_key_value_heads = 4

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config = bnb_config,
            trust_remote_code = True,
            config =config,
            )
        model.config.use_cache = False
        lora_alpha = 16
        lora_dropout = 0.1
        lora_r = 64   #

        modules = find_all_linear_names(model)
        peft_config = LoraConfig(
            lora_alpha = lora_alpha,
            lora_dropout= lora_dropout,
            r=lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=modules ,
        )
        model = get_peft_model(model, peft_config)
        model.delete_adapter('default')
        model.model.load_adapter(f'Anonymous-TAALM/{adapter_file}', adapter_name='default') 
        model.model.set_adapter('default')
        

        for name, module in model.named_modules():
            if "norm" in name :#or 'embed' in name:
                module = module.to(torch.float32)

        if 'gpt' in model_name:  
            for name, param in model.named_parameters():
                if 'ln' in name:
                    param.data = param.data.to(torch.float32)

            model.transformer.wte.to(torch.float32) 

        for name, param in model.named_parameters():
            if 'lora' in name :
                param.requires_grad = True
            else:
                param.requires_grad = False


        if regul:
            model.model.lm_head = nn.Linear(model.config.hidden_size,1, dtype=torch.float32, bias=True).to('cuda')
            nn.init.normal_(model.model.lm_head.weight, mean=0.0, std=0.001)
            model.model.lm_head.bias = torch.tensor(4.).to('cuda')
        else:
            model.model.lm_head = nn.Linear(model.config.hidden_size,1, dtype=torch.float32, bias=False).to('cuda')
        model.model.lm_head.requires_grad = True
        if regul:
            model.model.lm_head.bias.requires_grad=True


        if 'llama' in model_name.lower():
            model.model.model.embed_tokens.to(torch.float32)

        
        if onepiece:
            if 'llama' in model_name.lower():
                model.model.model.embed_tokens.requires_grad=False
            return model
        else:
            gamma =  model.model.model.embed_tokens  # theta 1b 위해
            model.base_model.model.model.embed_tokens = nn.Identity()

            return model, gamma  # phi

    @classmethod
    def reboot_theta_model(cls,  model, adapter_file='qlora'): 
        torch.manual_seed(42)
        model.delete_adapter('default')
        model.model.load_adapter(f'Anonymous-TAALM/{adapter_file}', adapter_name='default') 
        model.model.set_adapter('default')
        model.model.config.use_cache = False

        for name, param in model.named_parameters():
            if 'lora' in name :
                param.requires_grad = True
        return model    
    
    @classmethod
    def reboot_phi_model(cls,model, regul=False, adapter_file='qlora'):
        torch.manual_seed(42)
        model.delete_adapter('default')
        model.model.load_adapter(f'Anonymous-TAALM/{adapter_file}', adapter_name='default') 
        model.model.set_adapter('default')
        model.model.config.use_cache = False   

        if regul:
            model.model.lm_head = nn.Linear(model.config.hidden_size,1, dtype=torch.float32, bias=True).to('cuda')
            model.model.lm_head.bias = torch.tensor(4.).to('cuda')            
        else:
            model.model.lm_head = nn.Linear(model.config.hidden_size,1, dtype=torch.float32, bias=False).to('cuda')  
               
        for name, param in model.named_parameters():
            if 'lora' in name :
                param.requires_grad = True
        model.model.lm_head.requires_grad=True
        if regul:
             model.model.lm_head.bias.requires_grad=True
        
        return model
    
    @staticmethod  
    def calcul_loss(gamma,
                    theta, 
                    inputs,
                    tt_weight_vector, 
                    tt_target_weight=True, 
                    tt_L2=True,
                    tt_L2_weight=0.01,
                    known_masking= False,
                    giveme_logprobs=False,
                    TD_label=False,
                    onepiece=False):
        if onepiece:
            temp_inputs= inputs.copy()
            temp_inputs['labels'] = temp_inputs['input_ids']

        else:
            # gamma.train()
            after_gamma = gamma(inputs.input_ids)
            temp_inputs = inputs.copy()
            temp_inputs['labels'] = temp_inputs['input_ids'] 
            temp_inputs['input_ids'] = after_gamma
        outputs = theta(**temp_inputs)  
        weight_vector = tt_weight_vector.squeeze(-1)
        ####### w_loss  #######
        log_probs = -nn.functional.log_softmax(outputs.logits[:,:-1,:], dim=-1)
        
        labels0 = torch.clamp(inputs['input_ids'][:,1:], min=0).unsqueeze(-1)
        nll_loss = log_probs.gather(dim=-1, index=labels0).squeeze(-1)
        mask0 = inputs['attention_mask'][:,:-1]
        if theta.config._name_or_path == 'gpt2':
            mask0 = inputs['attention_mask'][:,1:].to('cuda')
        
        if known_masking:
            with torch.no_grad():
                known_mask = (~(torch.argmin(log_probs, dim=-1) == labels0.squeeze(-1))).int().float()
            mask0 = mask0 * known_mask

        weight_vector = weight_vector * mask0
        nll_loss = nll_loss * mask0
        if TD_label:
            nll_loss = nll_loss * weight_vector
        w_loss = nll_loss * weight_vector  

        w_loss_each = torch.sum(w_loss, dim=-1) # loss for each sentence
        w_sum = weight_vector.sum(dim=-1)  # weight for each sentence

        assert w_sum.shape == w_loss_each.shape
        w_loss_sepa = torch.div(w_loss_each , w_sum + 1e-6) 
        w_loss = w_loss_sepa.mean()  # output scalar

        loss_origin = outputs["loss"]

        if tt_target_weight:
            loss = w_loss
            loss_origin = loss_origin.detach()
            w_loss_sepa = w_loss_sepa.detach()
        else:
            loss = loss_origin
            loss_origin = loss_origin.detach()
            w_loss_sepa = w_loss_sepa.detach()           

        if tt_L2==True:
            l2=0
            for name, param in theta.named_parameters():
                if param.requires_grad==True:
                    
                    l2 +=  torch.norm(param,p=2)
            loss = loss + tt_L2_weight *l2

        if giveme_logprobs:
            return loss, loss_origin, w_loss_sepa, nll_loss, log_probs
        else:
            return loss, loss_origin, w_loss_sepa, nll_loss

    @staticmethod
    def extract_param(model, param_list, name_list, type0):

        assert param_list == []
        for name, param in model.named_parameters():
            if param.requires_grad:
                name_list.append(name)

                if type0=='clone_detach':
                    param_list.append(param.clone().detach())
                elif type0 == 'assign':
                    param_list.append(param)
    
    @staticmethod
    def inject_param(model, param_list, name_list, type0):
        for name, param in zip(name_list, param_list): 
            names = name.split('.')
            module_name = '.'.join(names[:-1])  # Get module path except for the last element
            param_name = names[-1] 
            module = model
            
            for sub_name in module_name.split('.'):
                if sub_name:
                    module = getattr(module, sub_name)
            
            if type0=='clone':
                setattr(module, param_name, param.clone())
            elif type0 == 'detach_reqgrad':
                setattr(module, param_name, param.detach().requires_grad_())
            elif type0 == 'detach':
                setattr(module, param_name, param.detach())
            elif type0 == 'assign':
                setattr(module, param_name, param)
    
    

##### kadapter
class Llama2_kadapter(AutoModelForCausalLM):
    def __init__(self):
        AutoModelForCausalLM.__init__(self)

    @classmethod
    def init_gamma_theta(cls, onepiece=False, model_name='TinyLlama/TinyLlama-1.1B-Chat-v1.0',**kwargs)-> AutoModelForCausalLM:
        torch.manual_seed(42)
        model = Llama2LMHeadModel(model_name=model_name)

        return model


class Llama2_fullfinetune(AutoModelForCausalLM):
    def __init__(self):
        AutoModelForCausalLM.__init__(self)

    @classmethod
    def init_gamma_theta(cls, model_name='TinyLlama/TinyLlama-1.1B-Chat-v1.0', onepiece=True, **kwargs) ->AutoModelForCausalLM:

        bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",   # data type
        bnb_4bit_compute_dtype=torch.float16,  # compute data type
        )
        model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True,output_attentions=True,
        torch_dtype=torch.bfloat16
        )
        model.config.use_cache = False

        model.model.embed_tokens.weight.requires_grad=False

        return model
