from functools import partial
from typing import Dict, Union

import pytorch_lightning as pl
from einops import repeat
from torch import Tensor, einsum, nn
from torch.nn import functional as F

from src.models.components import RetrieverBase
from src.models.components.probabilistic_embedding import ProbabilisticEmbedding


class ColBERT(RetrieverBase):
    def __init__(self, bert_name_or_module: Union[str, pl.LightningModule], similarity_fn: str):
        super().__init__(
            bert_name_or_module=bert_name_or_module,
            similarity_fn=similarity_fn,
            representation="multi",
        )
        self.embedding_dim = self.bert.embeddings.word_embeddings.embedding_dim
        self.linear = nn.Linear(self.embedding_dim, self.embedding_dim)
        # self.output_norm = nn.LayerNorm(embedding_dim)
        self.output_norm = partial(F.normalize, p=2, dim=-1)

    def set_validation(self):
        super().set_validation()
        self.similarity_fn = self.similarity_fns["cosine"]

    def encode_query(self, query):
        output = self.bert(query["input_ids"], query["attention_mask"])["last_hidden_state"]
        output = self.linear(output)
        output *= repeat(query["skip_mask"], "batch query_len -> batch query_len tmp", tmp=1)
        output = self.output_norm(output)
        final_output = query.copy()
        final_output.update({"encoded_embeddings": output})
        return final_output

    def encode_passage(self, passage):
        output = self.bert(passage["input_ids"], passage["attention_mask"])["last_hidden_state"]
        output = self.linear(output)
        output *= repeat(passage["skip_mask"], "batch doc_len -> batch doc_len tmp", tmp=1)
        output = self.output_norm(output)
        final_output = passage.copy()
        final_output.update({"encoded_embeddings": output})
        return final_output


class PEQColBERT(ColBERT):
    def __init__(
        self,
        bert_name_or_module: Union[str, pl.LightningModule],
        similarity_fn: str,
        n_samples: int,
        firstk: int,
    ):
        super().__init__(bert_name_or_module=bert_name_or_module, similarity_fn=similarity_fn)
        self.pe_layer = ProbabilisticEmbedding(
            in_dim=self.embedding_dim,
            out_dim=self.embedding_dim,
            n_samples=n_samples,
            firstk=firstk,
        )

    def set_validation(self):
        super().set_validation()
        self.pe_layer.n_samples = 0

    def encode_query(self, query):
        output = self.bert(query["input_ids"], query["attention_mask"])["last_hidden_state"]
        output = self.linear(output)
        output *= repeat(query["skip_mask"], "batch query_len -> batch query_len tmp", tmp=1)
        output = self.output_norm(output)
        # output = repeat(output, "batch query_len vocab -> batch tmp query_len vocab", tmp=1) \
        #          + self.pe_layer(output)
        pe_output = self.pe_layer(output)
        pe_output = self.output_norm(pe_output)
        final_output = query.copy()
        final_output.update({"encoded_embeddings": pe_output, "before_pe_embeddings": output})
        return final_output

    def encode_passage(self, passage):
        output = self.bert(passage["input_ids"], passage["attention_mask"])["last_hidden_state"]
        output = self.linear(output)
        output *= repeat(passage["skip_mask"], "batch doc_len -> batch doc_len tmp", tmp=1)
        output = self.output_norm(output)
        # output = output + self.pe_layer.mu_layer(output)
        pe_output = self.pe_layer.mu_layer(output)
        pe_output = self.output_norm(pe_output)
        final_output = passage.copy()
        final_output.update({"encoded_embeddings": pe_output, "before_pe_embeddings": output})
        return final_output

    @staticmethod
    def pe_residual_block(before_pe: Dict[str, Tensor], after_pe: Dict[str, Tensor]):
        assert all(map(lambda i: i in after_pe.keys(), before_pe.keys()))
        keys = before_pe.keys()
        return dict(map(lambda k: (k, before_pe[k] + after_pe[k]), keys))

    def squared_euclidian_score(
        self, query: Dict[str, Tensor], passage: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        org_query, pe_query = query["before_pe_embeddings"], query["encoded_embeddings"]
        org_passage, pe_passage = passage["before_pe_embeddings"], passage["encoded_embeddings"]
        self.score_input_check(org_query, org_passage)

        org_rels = super().squared_euclidian_score(
            {"encoded_embeddings": org_query}, {"encoded_embeddings": org_passage}
        )

        query_extended = repeat(
            pe_query, "batch sample query vocab -> batch sample query tmp vocab", tmp=1
        )
        passage_extended = repeat(
            pe_passage, "batch doc vocab -> batch tmp1 tmp2 doc vocab", tmp1=1, tmp2=1
        )

        pe_term_relevance: Tensor = -1.0 * ((query_extended - passage_extended) ** 2).sum(
            dim=-1
        ).mean(dim=1)
        pe_rels = self.term_relevance_to_final(pe_term_relevance)

        return self.pe_residual_block(org_rels, pe_rels)

    def cosine_score(
        self, query: Dict[str, Tensor], passage: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        org_query, pe_query = query["before_pe_embeddings"], query["encoded_embeddings"]
        org_passage, pe_passage = passage["before_pe_embeddings"], passage["encoded_embeddings"]
        self.score_input_check(org_query, org_passage)

        org_rels = super().cosine_score(
            {"encoded_embeddings": org_query}, {"encoded_embeddings": org_passage}
        )

        pe_term_relevance = einsum("bsqv,bdv->bsqd", pe_query, pe_passage).mean(dim=1)
        pe_rels = self.term_relevance_to_final(pe_term_relevance)

        return self.pe_residual_block(org_rels, pe_rels)

    def in_batch_score(
        self, query: Dict[str, Tensor], passage: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        org_query, pe_query = query["before_pe_embeddings"], query["encoded_embeddings"]
        org_passage, pe_passage = passage["before_pe_embeddings"], passage["encoded_embeddings"]
        self.score_input_check(org_query, org_passage, in_batch=True)

        org_term_relevance = einsum("aqv,bdv->abqd", org_query, org_passage)
        org_rels = self.term_relevance_to_final(org_term_relevance)

        pe_term_relevance = einsum("asqv,bdv->asbqd", pe_query, pe_passage).mean(dim=1)
        pe_rels = self.term_relevance_to_final(pe_term_relevance)

        return self.pe_residual_block(org_rels, pe_rels)


if __name__ == "__main__":
    colbert = ColBERT("distilbert-base-uncased")
