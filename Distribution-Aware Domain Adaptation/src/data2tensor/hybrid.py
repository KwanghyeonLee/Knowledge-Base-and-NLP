from typing import Dict, List, Union, Tuple

import torch
from torch import Tensor
from transformers import BatchEncoding

from src.data2tensor.tokenizer import BaseTokenizer


class HybridTokenizer:
    """This tokenizer does not make skip-mask for passage."""

    def __init__(self, dense: BaseTokenizer, sparse: BaseTokenizer):
        self.dense = dense
        self.sparse = sparse

    def tokenize_query(
            self, query: List[str]
    ) -> Tuple[Union[BatchEncoding, Dict[str, Tensor]], Union[BatchEncoding, Dict[str, Tensor]]]:
        dense_query = self.dense.tokenize_query(query)
        sparse_query = self.sparse.tokenize_query(query)

        return dense_query, sparse_query
    def tokenize_passage(
            self, passage: List[str]
    ) -> Tuple[Union[BatchEncoding, Dict[str, Tensor]], Union[BatchEncoding, Dict[str, Tensor]]]:
        dense_passage = self.dense.tokenize_passage(passage)
        sparse_passage = self.sparse.tokenize_passage(passage)
        return dense_passage, sparse_passage
