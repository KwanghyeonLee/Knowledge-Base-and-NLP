from typing import Dict, List, Optional, Union

import pytorch_lightning as pl
import torch
import transformers
from einops import repeat
from torch import Tensor, einsum, nn
from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM


class RetrieverBase(nn.Module):
    def encode(
        self,
        query: Dict[str, Tensor],
        passage1: Dict[str, Tensor],
        passage2: Optional[Dict[str, Tensor]] = None,
    ):
        if passage2 is not None:
            return (
                self.encode_query(query),
                self.encode_passage(passage1),
                self.encode_passage(passage2),
            )
        return self.encode_query(query), self.encode_passage(passage1)

    def encode_query(self, query: Dict[str, Tensor]) -> Dict[str, Tensor]:
        raise NotImplementedError

    def encode_passage(self, passage: Dict[str, Tensor]) -> Dict[str, Tensor]:
        raise NotImplementedError

    def set_validation(self) -> None:
        """This method sets retriever as validation mode."""
        self.training = False

    def __init__(
        self,
        bert_name_or_module: Union[str, pl.LightningModule],
        representation: str,
        similarity_fn: str,
        mlm: bool = False,
    ):
        # Retriever always use bert on base.
        super().__init__()

        # setting retriever's representation mode (either multi or single)
        if representation not in ["multi", "single"]:
            raise ValueError(
                f"representation must be either `multi` or `single`. (current: {representation}"
            )
        self.representation = representation

        # loading bert from given argument.
        # if it is given as pl.LightningModule then it takes student's bert.
        # else it must be loaded from huggingface.
        if issubclass(type(bert_name_or_module), pl.LightningModule):
            self.__class__ = bert_name_or_module.model.__class__
            self.__dict__ = bert_name_or_module.model.__dict__
            # self = bert_name_or_module.model
            # self.linear = bert_name_or_module.model.linear
            # if mlm:
            #     self.mlm = bert_name_or_module.model.mlm
            print("Loaded student's bert model")
        else:
            config = AutoConfig.from_pretrained(bert_name_or_module)
            if mlm:
                self.bert = AutoModelForMaskedLM.from_pretrained(bert_name_or_module, config=config)
            else:
                self.bert = AutoModel.from_pretrained(bert_name_or_module, config=config)

        self.embedding_dim = self.bert.config.hidden_size

        # setting similarity function
        self.similarity_fns = {
            "l2": self.squared_euclidian_score,
            "cosine": self.cosine_score,
            "dot": self.dot_score,
        }
        similarity_fn = similarity_fn.lower()
        if similarity_fn not in self.similarity_fns:
            raise ValueError(f"similarity function not supported ({similarity_fn})")
        self.similarity_fn = self.similarity_fns[similarity_fn]

    @staticmethod
    def score_input_check(query: Tensor, passage: Tensor, in_batch: bool = False):
        assert query.shape[-1] == passage.shape[-1]
        if in_batch:
            assert query.shape[0] * 2 == passage.shape[0]
        else:
            assert query.shape[0] == passage.shape[0]

    @staticmethod
    def term_relevance_to_final(term_relevance: Tensor) -> Dict[str, Tensor]:
        """This method only used for Multi-Vector Model.

        :param term_relevance: (batch_size, query_len, doc_len)
        :return:
            term_relevance: (batch_size, query_len, doc_len)
            query_maxsim: (batch_size, query_len)
            relevance: (batch_size)
        """
        query_maxsim = term_relevance.max(dim=-1).values
        relevance = query_maxsim.sum(dim=-1)
        return {
            "term_relevance": term_relevance,
            "query_maxsim": query_maxsim,
            "relevance": relevance,
        }

    def dot_score(self, query: Dict[str, Tensor], passage: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """This method only available for Single-Vector model.

        :param query: (batch_size, vocab_size)
        :param passage: (batch_size, vocab_size)
        :return: (batch, query_len, doc_len)
        """
        q, p = query["encoded_embeddings"], passage["encoded_embeddings"]
        self.score_input_check(q, p)
        return {"relevance": (q * p).sum(dim=-1)}

    def squared_euclidian_score(
        self, query: Dict[str, Tensor], passage: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        """This method only available for Multi-Vector model.

        :param query: (batch_size, query_len, vocab_size)
        :param passage: (batch_size, doc_len, vocab_size)
        :return: (batch, query_len, doc_len)
        """
        q, p = query["encoded_embeddings"], passage["encoded_embeddings"]
        self.score_input_check(q, p)
        query_extended = repeat(q, "batch query vocab -> batch query tmp vocab", tmp=1)
        passage_extended = repeat(p, "batch doc vocab -> batch tmp doc vocab", tmp=1)

        term_relevance: Tensor = -1.0 * ((query_extended - passage_extended) ** 2).sum(dim=-1)
        return self.term_relevance_to_final(term_relevance)

    def cosine_score(
        self, query: Dict[str, Tensor], passage: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        """This method only available for Multi-Vector model.

        :param query: (batch_size, query_len, vocab_size)
        :param passage: (batch_size, doc_len, vocab_size)
        :return: (batch, query_len, doc_len)
        """
        q, p = query["encoded_embeddings"], passage["encoded_embeddings"]
        self.score_input_check(q, p)
        if self.representation == "multi":
            term_relevance: Tensor = einsum("bqv,bdv->bqd", q, p)
            return self.term_relevance_to_final(term_relevance)
        if self.representation == "single":
            relevance: Tensor = einsum("bv,bv->b", q, p)
            return {"relevance": relevance}

    def in_batch_score(
        self, query: Dict[str, Tensor], passage: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        """This method only available for Multi-Vector model's in batch scoring.

        :param query: (batch_size, query_len, vocab_size)
        :param passage: (batch_size * 2, doc_len, vocab_size)
        :return: (batch, batch_size * 2, query_len, doc_len)
        """
        q, p = query["encoded_embeddings"], passage["encoded_embeddings"]
        self.score_input_check(q, p, in_batch=True)
        if self.representation == "multi":
            term_relevance: Tensor = einsum("aqv,bdv->abqd", q, p)
            return self.term_relevance_to_final(term_relevance)
        if self.representation == "single":
            relevance: Tensor = einsum("av,bv->ab", q, p)
            return {"relevance": relevance}

    def score(
        self,
        query: Dict[str, Tensor],
        passage: Union[List[Dict[str, Tensor]], Dict[str, Tensor]],
        in_batch: bool = False,
    ) -> Dict[str, Tensor]:
        """This method is called from retrieval_module implementing pl.LightningModule.

        :param query: encoded query embedding (normally, 'encoded_ids')
        :param passage: encoded passage embedding (normally, 'encoded_ids')
        :param in_batch: whether to do in-batch scoring or not.
        :return: dictionary of score related values
            this must include 'score' and may include 'term_relevance' and 'query_maxsim'.
        """
        if in_batch:
            # in-batch only provides cosine score
            # (batch_size * 2, doc_len, vocab_size)
            in_batch_passage: Dict[str, Tensor] = dict(
                map(
                    lambda key: (
                        key,
                        torch.cat([passage[0][key], passage[1][key]], dim=0)
                        if passage[0][key] is not None
                        else None,
                    ),
                    passage[0].keys(),
                )
            )
            return self.in_batch_score(query, in_batch_passage)

        return self.similarity_fn(query, passage)
