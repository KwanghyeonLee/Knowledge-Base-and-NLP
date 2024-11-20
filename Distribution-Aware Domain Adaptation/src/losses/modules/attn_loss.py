from typing import Dict

import torch
from einops import rearrange
from torch import Tensor, nn


class PositiveAttentionGuidedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.KLDivLoss(reduction="batchmean")

    def forward(self, embeddings: Dict[str, Tensor], teacher_ib_term_relevance: Tensor):
        device = embeddings["encoded_embeddings"].device
        _pos_sel = torch.arange(embeddings["encoded_embeddings"].size(0), device=device)

        # selecting query - positive calculation only
        term_rel_pos = teacher_ib_term_relevance[
            _pos_sel, _pos_sel, :, :
        ]  # (bsize, query_maxlen, doc_maxlen)
        max_sim = term_rel_pos.max(-1)  # (bsize, query_maxlen)

        # compute doc_attn loss
        doc_attn_label = torch.zeros(*tuple(embeddings["pooling_logit"].size()), device=device)
        pooled_idxs = torch.ones(*tuple(embeddings["pooling_logit"].size()), device=device)
        doc_attn_label.scatter_add_(dim=1, index=max_sim.indices, src=pooled_idxs)

        doc_attn_loss = self.loss_fn(
            embeddings["pooling_logit"].log_softmax(dim=1),
            doc_attn_label.masked_fill(embeddings["attention_mask"] == 0, -1e9).softmax(dim=1),
        )
        return doc_attn_loss


class TopKPositiveAttentionGuidedLoss(nn.Module):
    def __init__(self, topk: int = 5):
        super().__init__()
        self.topk = topk
        self.loss_fn = nn.KLDivLoss(reduction="batchmean")

    def forward(self, embeddings: Dict[str, Tensor], teacher_ib_term_relevance: Tensor):
        device = embeddings["encoded_embeddings"].device
        _pos_sel = torch.arange(embeddings["encoded_embeddings"].size(0), device=device)

        # selecting query - positive calculation only
        term_rel_pos = teacher_ib_term_relevance[
            _pos_sel, _pos_sel, :, :
        ]  # (bsize, query_maxlen, doc_maxlen)
        max_sim = term_rel_pos.topk(dim=-1, k=self.topk)  # (bsize, query_maxlen)
        max_sim = rearrange(max_sim.indices, "batch len topk -> batch (len topk)")

        # compute doc_attn loss
        doc_attn_label = torch.zeros(*tuple(embeddings["pooling_logit"].size()), device=device)
        pooled_idxs = torch.ones(*tuple(embeddings["pooling_logit"].size()), device=device)
        doc_attn_label.scatter_add_(dim=1, index=max_sim, src=pooled_idxs)

        doc_attn_loss = self.loss_fn(
            embeddings["pooling_logit"].log_softmax(dim=1),
            doc_attn_label.masked_fill(embeddings["attention_mask"] == 0, -1e9).softmax(dim=1),
        )
        return doc_attn_loss
