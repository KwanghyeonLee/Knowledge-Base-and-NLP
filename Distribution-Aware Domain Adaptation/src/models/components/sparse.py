from functools import partial
from typing import Optional, Union

import pytorch_lightning as pl
import torch
from einops import repeat
from torch import nn, log, relu
from torch.nn import functional as F
from transformers import AutoConfig, AutoModelForMaskedLM, BertForMaskedLM

from src.models.components import RetrieverBase


class SimpleSparse(RetrieverBase):
    def __init__(
        self,
        bert_name_or_module: Union[str, pl.LightningModule],
        similarity_fn: str,
    ):
        super().__init__(
            bert_name_or_module=bert_name_or_module,
            similarity_fn=similarity_fn,
            representation="single",
            mlm=True,
        )

    def encode_query(self, query):
        raise NotImplementedError

    def encode_passage(self, passage):
        raise NotImplementedError

class SPLADE(SimpleSparse):
    def __init__(self, bert_name_or_module: Union[str, pl.LightningModule], similarity_fn: str):
        super().__init__(bert_name_or_module, similarity_fn)
        self.output_org = lambda out: log(1 + relu(out)).amax(dim=1)

    @staticmethod
    def _normalize(tensor, eps=1e-9):
        """normalize input tensor on last dimension
        """
        return tensor / (torch.norm(tensor, dim=-1, keepdim=True) + eps)
    def encode_query(self, query):
        logits = self.bert(query["input_ids"], query["attention_mask"])["logits"]
        output = self.output_org(logits)
        output = self._normalize(output)

        final_output = query.copy()
        final_output.update({
            "encoded_logits": logits,
            "encoded_embeddings": output
        })
        return final_output

    def encode_passage(self, passage):
        logits = self.bert(passage["input_ids"], passage["attention_mask"])["logits"]
        output = self.output_org(logits)
        output = self._normalize(output)

        final_output = passage.copy()
        final_output.update({
            "encoded_logits": logits,
            "encoded_embeddings": output
        })
        return final_output


class SimpleSPLADE(SPLADE):
    def __init__(self, bert_name_or_module: Union[str, pl.LightningModule], similarity_fn: str):
        super().__init__(bert_name_or_module, similarity_fn)

    def encode_query(self, query):
        logits = self.bert(query["input_ids"], query["attention_mask"])["logits"]
        output = self.output_org(logits)
        output = self._normalize(output)

        final_output = query.copy()
        final_output.update({
            "encoded_embeddings": output
        })
        return final_output

    def encode_passage(self, passage):
        logits = self.bert(passage["input_ids"], passage["attention_mask"])["logits"]
        output = self.output_org(logits)
        output = self._normalize(output)

        final_output = passage.copy()
        final_output.update({
            "encoded_embeddings": output
        })
        return final_output

class MLMHead(nn.Module):
    def __init__(self, bert_name_or_module: Union[str, pl.LightningModule]):
        super().__init__()

        config = AutoConfig.from_pretrained(bert_name_or_module)
        bert: BertForMaskedLM = AutoModelForMaskedLM.from_pretrained(bert_name_or_module, config=config)
        self.mlm = bert.cls
        self.output_org = lambda out: log(1 + relu(out)).amax(dim=1)

    @staticmethod
    def _normalize(tensor, eps=1e-9):
        """normalize input tensor on last dimension
        """
        return tensor / (torch.norm(tensor, dim=-1, keepdim=True) + eps)

    def forward(self, embeddings):
        logits = self.mlm(embeddings)
        output = self.output_org(logits)
        output = self._normalize(output)
        return {
            "encoded_logits": logits,
            "encoded_embeddings": output
        }




