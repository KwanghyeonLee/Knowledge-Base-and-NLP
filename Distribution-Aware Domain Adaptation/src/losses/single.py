from typing import Dict

import torch
from torch import Tensor, nn
from torch.nn import NLLLoss

from src.losses.modules import MultipleObjectives, SingleObjective
from src.losses.modules.ib_crossentropy import InBatchCrossEntropyLoss
from src.losses.modules.margin_mse import MarginMSELoss
from src.losses.modules.similarity_loss import (
    EmbeddingNegativeCosineSimilarityLoss,
)


class MarginDistillationLoss(MultipleObjectives):
    def __init__(self):
        super().__init__()
        self.scales = {"margin_mse": 1}
        self.margin_mse = MarginMSELoss(scale=self.scales["margin_mse"])

    def forward(
            self,
            query_emb: Dict[str, Tensor],
            pos_pas_emb: Dict[str, Tensor],
            neg_pas_emb: Dict[str, Tensor],

            teacher_query_emb: Dict[str, Tensor],
            teacher_pos_pas_emb: Dict[str, Tensor],
            teacher_neg_pas_emb: Dict[str, Tensor],

            pos_score_res: Dict[str, Tensor],
            neg_score_res: Dict[str, Tensor],

            label: Dict[str, Tensor],
            *args, **kwargs
    ):
        # gpl guided loss
        pred_pos, pred_neg = pos_score_res["relevance"], neg_score_res["relevance"]
        margin_mse_loss = self.margin_mse(pred_pos, pred_neg, label)

        losses = {
            "student_loss": margin_mse_loss,
        }
        losses["loss"] = losses["student_loss"]
        return losses


class MarginDistillationOnlyOnline(MultipleObjectives):
    def __init__(self):
        super().__init__()
        self.scales = {"margin_mse": 1}
        self.margin_mse = MarginMSELoss(scale=self.scales["margin_mse"])

    def forward(
        self,
        query_emb,
        pos_pas_emb,
        neg_pas_emb,
        teacher_query_emb,
        teacher_pos_pas_emb,
        teacher_neg_pas_emb,
        pos_score_res,
        neg_score_res,
        ib_score_res,
        soft_ib_res,
        label,
    ):
        pred_pos, pred_neg = pos_score_res["relevance"], neg_score_res["relevance"]
        pred_ib, soft_label_ib = ib_score_res["relevance"], soft_ib_res["relevance"]

        batch_size = soft_label_ib.shape[0]
        pos_soft_label = soft_label_ib.diagonal()
        neg_soft_label = soft_label_ib.diagonal(offset=batch_size)
        margin_soft_label = pos_soft_label - neg_soft_label

        margin_mse_loss = self.margin_mse(pred_pos, pred_neg, margin_soft_label)
        losses = {"student_loss": margin_mse_loss}
        losses["loss"] = losses["student_loss"]
        return losses


class InBatchContrastiveLoss(MultipleObjectives):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.nll = NLLLoss(reduction="mean")
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.margin_mse = MarginMSELoss(scale=1)
        self.cross_entropy = nn.CrossEntropyLoss(reduction="mean")

    def forward(
        self,
        query_emb,
        pos_pas_emb,
        neg_pas_emb,
        teacher_query_emb,
        teacher_pos_pas_emb,
        teacher_neg_pas_emb,
        pos_score_res,
        neg_score_res,
        ib_score_res,
        soft_ib_res,
        label,
    ):
        ib_score_res = ib_score_res["relevance"]

        batch_size = ib_score_res.shape[0]
        ib_score_res = torch.exp(ib_score_res)
        pos_scores = ib_score_res.diagonal()
        neg_scores = ib_score_res[~torch.eye(*ib_score_res.shape).bool()].reshape(
            batch_size, batch_size - 1
        ).sum(dim=-1)

        contrastive_nll_loss = - torch.log(pos_scores / (pos_scores + neg_scores)).mean()

        # softmax_score = self.log_softmax(ib_score_res)
        # nll_loss = self.nll(softmax_score, torch.arange(softmax_score.shape[0], device=softmax_score.device))
        query_vec = query_emb["encoded_embeddings"]
        doc_vec = pos_pas_emb["encoded_embeddings"]

        cos_sim = nn.functional.cosine_similarity(query_vec, doc_vec, dim=-1)
        ce_loss = self.cross_entropy(cos_sim, label.float())

        losses = {"ce_loss": ce_loss, "contrast_loss": contrastive_nll_loss}
        losses["loss"] = losses["ce_loss"] + losses["contrast_loss"]
        return losses
