from functools import partial
from typing import Optional, Union

import pytorch_lightning as pl
from einops import repeat
from torch import nn
from torch.nn import functional as F

from src.models.components import RetrieverBase
from src.models.components.pooling import (
    AdaptivePooling,
    AttentionDropoutPooling,
    AttentionPooling,
    BasePooling,
    LSTMPooling,
    UNetPooling,
)


class NaiveDense(RetrieverBase):
    def __init__(
        self,
        bert_name_or_module: Union[str, pl.LightningModule],
        similarity_fn: str,
        pooling_method: Optional[str],
    ):
        super().__init__(
            bert_name_or_module=bert_name_or_module,
            similarity_fn=similarity_fn,
            representation="single",
        )
        self.linear = nn.Linear(self.embedding_dim, self.embedding_dim)
        # self.output_norm = nn.LayerNorm(embedding_dim)
        self.output_norm = partial(F.normalize, p=2, dim=-1)
        if pooling_method:
            self.pooling = BasePooling(pooling_method)

    def encode_query(self, query):
        output = self.bert(query["input_ids"], query["attention_mask"])["last_hidden_state"]
        output = self.linear(output)
        output = output * repeat(query["skip_mask"], "batch query_len -> batch query_len tmp", tmp=1)
        # output = self.output_norm(output)
        output, pool_logit = self.pooling(output, attention_mask=query["attention_mask"])
        final_output = query.copy()
        final_output.update({
            "encoded_embeddings": output
        })
        return final_output

    def encode_passage(self, passage):
        output = self.bert(passage["input_ids"], passage["attention_mask"])["last_hidden_state"]
        output = self.linear(output)
        output = output * repeat(passage["skip_mask"], "batch doc_len -> batch doc_len tmp", tmp=1)
        # output = self.output_norm(output)
        output, pool_logit = self.pooling(output, attention_mask=passage["attention_mask"])
        final_output = passage.copy()
        final_output.update({
            "encoded_embeddings": output
        })
        return final_output


class SimpleDense(RetrieverBase):
    def __init__(
        self,
        bert_name_or_module: Union[str, pl.LightningModule],
        similarity_fn: str,
        pooling_method: Optional[str],
        # is_linear: bool = True,
    ):
        super().__init__(
            bert_name_or_module=bert_name_or_module,
            similarity_fn=similarity_fn,
            representation="single",
        )
        # self.is_linear = is_linear
        # if is_linear:
        self.linear = nn.Linear(self.embedding_dim, self.embedding_dim)
        # self.output_norm = nn.LayerNorm(embedding_dim)
        self.output_norm = partial(F.normalize, p=2, dim=-1)
        if pooling_method:
            self.pooling = BasePooling(pooling_method)

    def encode_query(self, query):
        self.bert._use_flash_attention_2 = False
        output = self.bert(query["input_ids"], query["attention_mask"])["last_hidden_state"]
        # if self.is_linear:
        output_mat = self.linear(output)
        # else:
        #     output_mat = output
        output = output_mat * repeat(query["skip_mask"], "batch query_len -> batch query_len tmp", tmp=1)
        # output = self.output_norm(output)
        output, pool_logit = self.pooling(output, attention_mask=query["attention_mask"])
        final_output = query.copy()
        final_output.update({
            "encoded_matrix": output_mat,
            "encoded_embeddings": output,
            # "pooling_logit": pool_logit
        })
        return final_output

    def encode_passage(self, passage):
        self.bert._use_flash_attention_2 = False
        output = self.bert(passage["input_ids"], passage["attention_mask"])["last_hidden_state"]
        # if self.is_linear:
        output_mat = self.linear(output)
        # else:
        #     output_mat = output
        output = output_mat * repeat(passage["skip_mask"], "batch doc_len -> batch doc_len tmp", tmp=1)
        # output = self.output_norm(output)
        output, pool_logit = self.pooling(output, attention_mask=passage["attention_mask"])
        final_output = passage.copy()
        final_output.update({
            "encoded_matrix": output_mat,
            "encoded_embeddings": output,
            # "pooling_logit": pool_logit
        })
        return final_output


class DenseT5(SimpleDense):
    def __init__(self, bert_name_or_module: Union[str, pl.LightningModule], similarity_fn: str, pooling_method: Optional[str]):
        super().__init__(bert_name_or_module, similarity_fn, pooling_method)
        self.bert = self.bert.get_encoder()


class MLMDense(RetrieverBase):
    def __init__(self, bert_name_or_module: Union[str, pl.LightningModule], similarity_fn: str, pooling_method: Optional[str]):
        super().__init__(bert_name_or_module, similarity_fn, pooling_method, mlm=True)


class DenseMLM(MLMDense):
    def __init__(self, bert_name_or_module: Union[str, pl.LightningModule], similarity_fn: str, pooling_method: Optional[str]):
        super().__init__(bert_name_or_module, "single", similarity_fn)

        if not issubclass(type(bert_name_or_module), pl.LightningModule):
            self.mlm = self.bert.cls
            self.bert = self.bert.bert

            self.linear = nn.Linear(self.embedding_dim, self.embedding_dim)
            if pooling_method:
                self.pooling = BasePooling(pooling_method)

    def encode_query(self, query):
        output = self.bert(query["input_ids"], query["attention_mask"])["last_hidden_state"]
        output_mat = self.linear(output)
        output = output_mat * repeat(query["skip_mask"], "batch query_len -> batch query_len tmp", tmp=1)
        # output = self.output_norm(output)
        output, pool_logit = self.pooling(output, attention_mask=query["attention_mask"])
        output_vocab = self.mlm(output)

        final_output = query.copy()
        final_output.update({
            "encoded_embeddings": output,
            "encoded_vocabs": output_vocab
            # "pooling_logit": pool_logit
        })
        return final_output

    def encode_passage(self, passage):
        output = self.bert(passage["input_ids"], passage["attention_mask"])["last_hidden_state"]
        output_mat = self.linear(output)
        output = output_mat * repeat(passage["skip_mask"], "batch doc_len -> batch doc_len tmp", tmp=1)
        # output = self.output_norm(output)
        output, pool_logit = self.pooling(output, attention_mask=passage["attention_mask"])
        output_vocab = self.mlm(output)

        final_output = passage.copy()
        final_output.update({
            "encoded_embeddings": output,
            "encoded_vocabs": output_vocab
            # "pooling_logit": pool_logit
        })
        return final_output


class ATDSDense(RetrieverBase):
    def __init__(
        self,
        bert_name_or_module: Union[str, pl.LightningModule],
        similarity_fn: str,
        pooling_method: Optional[str],
    ):
        super().__init__(
            bert_name_or_module=bert_name_or_module,
            similarity_fn=similarity_fn,
            representation="single",
        )
        self.linear = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.weight = nn.Linear(self.embedding_dim, 1)
        # self.output_norm = nn.LayerNorm(embedding_dim)
        self.output_norm = partial(F.normalize, p=2, dim=-1)
        if pooling_method:
            self.pooling = BasePooling(pooling_method)

    def encode_query(self, query):
        output = self.bert(query["input_ids"], query["attention_mask"])["last_hidden_state"]
        output_mat = self.linear(output)
        output = output_mat * repeat(query["skip_mask"], "batch query_len -> batch query_len tmp", tmp=1)
        weight = self.weight(output)
        # output = self.output_norm(output)
        output, pool_logit = self.pooling(output, attention_mask=query["attention_mask"])
        final_output = query.copy()
        final_output.update({
            "weight": weight,
            "encoded_matrix": output_mat,
            "encoded_embeddings": output, "pooling_logit": pool_logit
        })
        return final_output

    def encode_passage(self, passage):
        output = self.bert(passage["input_ids"], passage["attention_mask"])["last_hidden_state"]
        output_mat = self.linear(output)
        output = output_mat * repeat(passage["skip_mask"], "batch doc_len -> batch doc_len tmp", tmp=1)
        weight = self.weight(output)
        # output = self.output_norm(output)
        output, pool_logit = self.pooling(output, attention_mask=passage["attention_mask"])
        final_output = passage.copy()
        final_output.update({
            "weight": weight,
            "encoded_matrix": output_mat,
            "encoded_embeddings": output, "pooling_logit": pool_logit
        })
        return final_output


class DenseAttn(SimpleDense):
    def __init__(self, bert_name_or_module: Union[str, pl.LightningModule], similarity_fn: str):
        super().__init__(bert_name_or_module, similarity_fn, None)
        self.pooling = AttentionPooling(self.embedding_dim)


class DenseAttnDropout(SimpleDense):
    def __init__(
        self,
        bert_name_or_module: Union[str, pl.LightningModule],
        similarity_fn: str,
        dropout: float,
    ):
        super().__init__(bert_name_or_module, similarity_fn, None)
        self.pooling = AttentionDropoutPooling(self.embedding_dim, dropout=dropout)


class DenseLSTM(SimpleDense):
    def __init__(self, bert_name_or_module: Union[str, pl.LightningModule], similarity_fn: str):
        super().__init__(bert_name_or_module, similarity_fn, None)
        self.pooling = LSTMPooling(self.embedding_dim)


class DenseAdaptive(SimpleDense):
    def __init__(self, bert_name_or_module: Union[str, pl.LightningModule], similarity_fn: str):
        super().__init__(bert_name_or_module, similarity_fn, None)
        self.pooling = AdaptivePooling(self.embedding_dim)


class DenseUNet(SimpleDense):
    def __init__(self, bert_name_or_module: Union[str, pl.LightningModule], similarity_fn: str):
        super().__init__(bert_name_or_module, similarity_fn, None)
        self.pooling = UNetPooling(self.embedding_dim)


class DualDense(RetrieverBase):
    def __init__(
        self,
        bert_name_or_module: Union[str, pl.LightningModule],
        similarity_fn: str,
        pooling_method: Optional[str],
    ):
        super().__init__(
            bert_name_or_module=bert_name_or_module,
            similarity_fn=similarity_fn,
            representation="single",
        )
        self.linear = nn.Linear(self.embedding_dim, self.embedding_dim)
        # self.output_norm = nn.LayerNorm(embedding_dim)
        self.output_norm = partial(F.normalize, p=2, dim=-1)
        if pooling_method:
            self.query_pooling = BasePooling(pooling_method)
            self.passage_pooling = BasePooling(pooling_method)

    def encode_query(self, query):
        output = self.bert(query["input_ids"], query["attention_mask"])["last_hidden_state"]
        output = self.linear(output)
        output *= repeat(query["skip_mask"], "batch query_len -> batch query_len tmp", tmp=1)
        # output = self.output_norm(output)
        output, pool_logit = self.query_pooling(output, attention_mask=query["attention_mask"])
        final_output = query.copy()
        final_output.update({"encoded_embeddings": output, "pooling_logit": pool_logit})
        return final_output

    def encode_passage(self, passage):
        output = self.bert(passage["input_ids"], passage["attention_mask"])["last_hidden_state"]
        output = self.linear(output)
        output *= repeat(passage["skip_mask"], "batch doc_len -> batch doc_len tmp", tmp=1)
        # output = self.output_norm(output)
        output, pool_logit = self.passage_pooling(output, attention_mask=passage["attention_mask"])
        final_output = passage.copy()
        final_output.update({"encoded_embeddings": output, "pooling_logit": pool_logit})
        return final_output


class DualDenseAttn(DualDense):
    def __init__(self, bert_name_or_module: Union[str, pl.LightningModule], similarity_fn: str):
        super().__init__(bert_name_or_module, similarity_fn, None)
        self.query_pooling = AttentionPooling(self.embedding_dim)
        self.passage_pooling = AttentionPooling(self.embedding_dim)
