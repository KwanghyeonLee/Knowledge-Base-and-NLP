from typing import Dict

from torch import Tensor, nn


class SingleObjective(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, pred_pos_result: Dict[str, Tensor], pred_neg_result: Dict[str, Tensor], label
    ):
        """This forward is used for model only has one objective.

        :param pred_pos_result: student model's query-positive_passage score
        :param pred_neg_result: student model's query-negative_passage score
        :param label: answer
        :return: {
            "loss": ...,
            "student_loss": ...
        }
            loss = student_loss
        """
        raise NotImplementedError


class MultipleObjectives(nn.Module):
    def __init__(self):
        super().__init__()

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
        ib_score_res: Dict[str, Tensor],
        soft_ib_res: Dict[str, Tensor],
        label: Dict[str, Tensor],
    ):
        """This forward is used for model has multiple objective. Mostly used for knowledge
        distillation.

        :param query_emb: student model's query embedding
        :param pos_pas_emb: student model's positive passage embedding
        :param neg_pas_emb: student model's negative passage embedding
        :param teacher_query_emb: teacher model's query embedding
        :param teacher_pos_pas_emb: teacher model's positive passage embedding
        :param teacher_neg_pas_emb: teacher model's negative passage embedding
        :param pos_score_res: student model's query - positive passage score
        :param neg_score_res: student model's query - negative passage score
        :param ib_score_res: student model's in-batch score result
        :param soft_ib_res: teacher model's in-batch score result
        :param label: answer
        :return: {
            "loss": ...,
            "student_loss": ...,
            "teacher_loss": ...,
            ...
        }
            loss = student_loss + teacher_loss + ...
        """
        raise NotImplementedError
