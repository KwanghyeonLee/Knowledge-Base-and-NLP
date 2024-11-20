import string
from typing import Dict, List, Union

import torch
from torch import Tensor
from transformers import AutoTokenizer, BatchEncoding, BertTokenizer, T5Tokenizer, DistilBertTokenizer, AutoConfig


class BaseTokenizer:
    def __init__(self, model_name_or_path: str, max_query_length: int, max_doc_length: int):
        self.max_query_length = max_query_length
        self.max_doc_length = max_doc_length
        self.max_seq_length = max_query_length if max_query_length == max_doc_length else max_doc_length

        if "t5" in model_name_or_path:
            self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(model_name_or_path)
        else:
            self.tokenizer: BertTokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        self.skiplist = {
            w: True
            for symbol in string.punctuation
            for w in [symbol, self.tokenizer.encode(symbol, add_special_tokens=False)[0]]
        }

    def tokenize(self, batch_text: List[str]):
        return self.tokenizer(
            batch_text,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_seq_length,
        )

    @staticmethod
    def mask(input_ids, skiplist):
        mask = [[(x not in skiplist) and (x != 0) for x in d] for d in input_ids.cpu().tolist()]
        return mask

    def make_human_readable(self, tensor: Tensor):
        return self.tokenizer.batch_decode(tensor)

    def padding_expansion(self, encoded_tokens: BatchEncoding):
        # sep -> mask, pad -> mask
        encoded_tokens["input_ids"][
            encoded_tokens["input_ids"] == self.tokenizer.sep_token_id
        ] = self.tokenizer.mask_token_id
        encoded_tokens["input_ids"][
            encoded_tokens["input_ids"] == self.tokenizer.pad_token_id
        ] = self.tokenizer.mask_token_id
        # last token -> sep
        encoded_tokens["input_ids"][:, -1] = self.tokenizer.sep_token_id

        # make all tokens to be activated.
        encoded_tokens["attention_mask"][:] = 1

        return encoded_tokens

    def tokenize_query(self, query: List[str]) -> Union[BatchEncoding, Dict[str, Tensor]]:
        query = self.tokenizer(
            query,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=self.max_query_length,
        )
        # query = self.padding_expansion(query)

        query["skip_mask"] = torch.tensor(self.mask(query["input_ids"], skiplist=[])).float()
        return query

    def tokenize_passage(self, passage: List[str]) -> Union[BatchEncoding, Dict[str, Tensor]]:
        passage = self.tokenizer(
            passage,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=self.max_doc_length,
        )
        passage["skip_mask"] = torch.tensor(
            self.mask(passage["input_ids"], skiplist=self.skiplist)
        ).float()
        return passage
