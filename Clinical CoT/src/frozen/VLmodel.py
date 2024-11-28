from typing import Dict, Optional
import torch
import torch.nn as nn
import transformers
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    LlamaTokenizerFast,
)
from src.frozen.resnet import generate_resnet_model

# class Frozen(nn.Module, Generation):
class Frozen(nn.Module):    
    def __init__(self, model_args):
        super(Frozen, self).__init__()

        self.model_args = model_args
        
        # Get Model
        self.language_model = AutoModelForCausalLM.from_pretrained(
            self.model_args.model_name_or_path,
            torch_dtype=torch.float32
        )
        
        # Get Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.model_name_or_path,
            padding_side="right",
            use_fast=True,
        )
        
        # initialize the pad token
        if self.tokenizer._pad_token is None:
            if isinstance(self.tokenizer, LlamaTokenizerFast):
                self.tokenizer._pad_token = self.tokenizer._unk_token
            else: # OPT model
                smart_tokenizer_and_embedding_resize(
                    special_tokens_dict=dict(pad_token="[PAD]"),
                    tokenizer=self.tokenizer,
                    model=self.model,
                )
        

        if self.model_args.freeze_LM:
            for param in self.language_model.parameters():
                param.requires_grad = False

        # initialize the resnet model
        self.vision_encoder = generate_resnet_model(
            self.model_args.resnet_depth, n_input_channels=self.model_args.img_input_channel, conv1_t_stride=2)
        
        # change the last layer of the resnet to match the gpt2 embedding size
        self.vision_encoder.fc = nn.Linear(
            self.model_args.resnet_fc_dim, self.language_model.config.hidden_size * self.model_args.img_token_num)

    def forward(self, input_ids, input_img: Optional[torch.LongTensor], **kwargs):
        if input_img is None:
            return self.language_model(**kwargs)

        kwargs['inputs_embeds'] = self.get_inputs_embeds(input_img, input_ids)
        return self.language_model(**kwargs)
                

    def get_inputs_embeds(self, input_img, input_ids):
        text_embeds = self.language_model.get_input_embeddings()(input_ids)
        vision_embeds = self.vision_encoder(input_img)
        vision_embeds = vision_embeds.reshape(
            -1, self.model_args.img_token_num, self.language_model.config.hidden_size)
        input_embeds = torch.cat((vision_embeds, text_embeds), dim=1)
        return input_embeds

    def get_tokenizer(self):
        return self.tokenizer
    
    def generate(self, input_img, input_ids, **kwargs):
        kwargs['inputs_embeds'] = self.get_inputs_embeds(input_img, input_ids)
        kwargs['bos_token_id'] = self.tokenizer.bos_token_id
        kwargs['pad_token_id'] = self.tokenizer.pad_token_id
        kwargs['eos_token_id'] = self.tokenizer.eos_token_id
        return self.language_model.generate(**kwargs)
    


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """
    Resize tokenizer and embeddings
    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg
    