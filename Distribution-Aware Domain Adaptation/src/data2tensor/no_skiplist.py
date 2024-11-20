from typing import Dict, List

import torch
from torch import Tensor

from src.data2tensor.tokenizer import BaseTokenizer


class NoSkipListTokenizer(BaseTokenizer):
    """This tokenizer does not make skip-mask for passage."""

    def __init__(self, model_name_or_path: str, max_query_length: int, max_doc_length: int):
        super().__init__(model_name_or_path, max_query_length, max_doc_length)

    def tokenize_passage(self, passage: List[str]) -> Dict[str, Tensor]:
        passage = self.tokenize(passage)
        passage["skip_mask"] = torch.tensor(self.mask(passage["input_ids"], skiplist=[])).float()
        return passage
