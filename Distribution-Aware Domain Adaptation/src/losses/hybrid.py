import json
from typing import Dict

import torch
from einops import einsum, repeat, reduce, rearrange
from torch import Tensor, nn, tensor
from torch.nn import MSELoss
from tqdm.rich import tqdm

from src.losses.modules.js_div import JSDivergence
from src.losses.modules.kl_div import KLDivergenceLoss
from src.losses.modules.margin_mse import MarginMSELoss


class Dense2SparseMapCriterion(nn.Module):
    def __init__(self, idf_path: str, map_method: str):
        super().__init__()
        self.scales = {"margin_mse": 1}
        self.margin_mse = MarginMSELoss(scale=self.scales["margin_mse"])
        self.kl_div = KLDivergenceLoss()

        # idf
        self.idf = tensor(json.load(open(idf_path, "r")))
        mean, std = torch.mean(self.idf), torch.std(self.idf)
        self.idf_norm = (self.idf - mean) / std

        # utils
        self.map_method = map_method

    def map_builder(self, dense: Tensor, sparse: Tensor) -> Tensor:
        output = einsum(
            dense, sparse,
            "batch len dense, batch len sparse -> batch dense sparse"
        )
        if self.map_method == "max":
            output = output.amax(dim=1)
        elif self.map_method == "mean":
            output = output.mean(dim=1)
        elif self.map_method == "sum":
            output = output.sum(dim=1)
        else:
            raise NotImplementedError
        return output

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

        # perpendicular between pos and neg?

        # custom loss
        # query_map = self.map_builder(
        #     query_emb["encoded_matrix"], teacher_query_emb["encoded_logits"],
        # )
        # query_map_loss = self.kl_div(query_map, self.idf_norm.to(query_map.device))

        pos_pas_map = self.map_builder(
            pos_pas_emb["encoded_matrix"], teacher_pos_pas_emb["encoded_logits"]
        )
        pos_pas_map_loss = self.kl_div(pos_pas_map, self.idf_norm.to(pos_pas_map.device))

        neg_pas_map = self.map_builder(
            neg_pas_emb["encoded_matrix"], teacher_neg_pas_emb["encoded_logits"]
        )
        neg_pas_map_loss = self.kl_div(neg_pas_map, self.idf_norm.to(neg_pas_map.device))

        # map_loss = query_map_loss + pos_pas_map_loss + neg_pas_map_loss
        map_loss = pos_pas_map_loss + neg_pas_map_loss

        # return
        losses = {"student_loss": margin_mse_loss, "map_loss": map_loss}
        losses["loss"] = losses["student_loss"] + losses["map_loss"]
        return losses


class Dense2SparseMapSemanticIDFCriterion(Dense2SparseMapCriterion):
    def __init__(self, idf_path: str, map_method: str):
        super().__init__(idf_path, map_method)

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

        device = pred_pos.device

        # idf.shape = [vocab_size], teacher_***_pas_emb["encoded_embeddings"].shape = [vocab_size]
        pos_pas_map = self.map_builder(
            pos_pas_emb["encoded_matrix"], teacher_pos_pas_emb["encoded_logits"]
        )
        pos_pas_semantic_idf = self.idf_norm.to(device) * teacher_pos_pas_emb["encoded_embeddings"]
        pos_pas_map_loss = self.kl_div(pos_pas_map, pos_pas_semantic_idf)

        neg_pas_map = self.map_builder(
            neg_pas_emb["encoded_matrix"], teacher_neg_pas_emb["encoded_logits"]
        )
        neg_pas_semantic_idf = self.idf_norm.to(device) * teacher_neg_pas_emb["encoded_embeddings"]
        neg_pas_map_loss = self.kl_div(neg_pas_map, neg_pas_semantic_idf)

        # map_loss = query_map_loss + pos_pas_map_loss + neg_pas_map_loss
        map_loss = pos_pas_map_loss + neg_pas_map_loss

        # return
        losses = {"student_loss": margin_mse_loss, "map_loss": map_loss}
        losses["loss"] = losses["student_loss"] + losses["map_loss"]
        return losses


class Dense2SparseMapSemanticIDFNormCriterion(Dense2SparseMapCriterion):
    def get_guide_normalizer(self, method: str):
        # aim: 0 ~ 2 && 0 ~ 8 => make this fit into same range
        # reference normalization methods
        # https://chat.openai.com/share/43ab949b-681e-4159-83a9-4e8e23b153eb

        # tanh with tiny eps looks best
        # primarily: Large val * Large val
        #   all values fall into [-1, 1] therefore, multiplication between two value cannot exceed 1
        #   this is important because we are using this as a weight
        #   if the weight is too large, it will dominate the loss
        # secondary: Small val * Small val
        #   The shape of tanh boosts small values
        #   in addition, because all values are in [-1, 1], the multiplication between two small values
        #   will have similar effect as large values
        # lastly: 0 * some value
        #   even the value is meaningful on one side, zero make it total zero.
        #   to avoid this, we add tiny eps
        # therefore, all values are not ignored (0 * sm) and not too large (lg * lg)

        eps = 1e-8
        normalizers = {
            "softmax": nn.Softmax(dim=-1),
            "softmax_eps": lambda x: nn.Softmax(dim=-1)(x + eps),
            "softmax_zscore": lambda x: nn.Softmax(dim=-1)(
                (x - torch.mean(x, dim=-1, keepdim=True)) / (torch.std(x, dim=-1, keepdim=True))
            ),
            "tanh": nn.Tanh(),
            "tanh_eps": lambda x: nn.Tanh()(x + eps),
            "tanh_zscore_eps": lambda x: nn.Tanh()(
                (x - torch.mean(x, dim=-1, keepdim=True)) / (torch.std(x, dim=-1, keepdim=True) + eps)
            ),
            "sigmoid": nn.Sigmoid(),
            "sigmoid_eps": lambda x: nn.Sigmoid()(x + eps),
            "sigmoid_zscore_eps": lambda x: nn.Sigmoid()(
                (x - torch.mean(x, dim=-1, keepdim=True)) / (torch.std(x, dim=-1, keepdim=True) + eps)
            ),
            "log": lambda x: torch.log(x + 1),
            "l1": lambda x: x / torch.norm(x, p=1, dim=-1, keepdim=True),
            "l2": lambda x: x / torch.norm(x, p=2, dim=-1, keepdim=True),
            "zscore": lambda x: (x - torch.mean(x, dim=-1, keepdim=True)) / (torch.std(x, dim=-1, keepdim=True) + eps),
            "minmax": (
                lambda x: (x - torch.min(x, dim=-1, keepdim=True)[0])
                          / (torch.max(x, dim=-1, keepdim=True)[0] - torch.min(x, dim=-1, keepdim=True)[0] + eps)
            ),

        }
        if method not in normalizers:
            raise NotImplementedError

        return normalizers[method]

    def __init__(self, idf_path: str, map_method: str, norm_method: str):
        super().__init__(idf_path, map_method)
        self.norm_method = self.get_guide_normalizer(norm_method)
        # self.mult_after_norm = self.get_guide_normalizer("zscore")

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

        device = pred_pos.device

        # idf.shape = [vocab_size], teacher_***_pas_emb["encoded_embeddings"].shape = [vocab_size]
        pos_pas_map = self.map_builder(
            dense=pos_pas_emb["encoded_matrix"], sparse=teacher_pos_pas_emb["encoded_logits"]
        )
        pos_pas_semantic_idf = (
                self.norm_method(self.idf.to(device))
                * self.norm_method(teacher_pos_pas_emb["encoded_embeddings"])
        )
        pos_pas_map_loss = self.kl_div(pos_pas_map, pos_pas_semantic_idf)

        neg_pas_map = self.map_builder(
            dense=neg_pas_emb["encoded_matrix"], sparse=teacher_neg_pas_emb["encoded_logits"]
        )
        neg_pas_semantic_idf = (
                self.norm_method(self.idf.to(device))
                * self.norm_method(teacher_neg_pas_emb["encoded_embeddings"])
        )
        neg_pas_map_loss = self.kl_div(neg_pas_map, neg_pas_semantic_idf)

        # map_loss = query_map_loss + pos_pas_map_loss + neg_pas_map_loss
        map_loss = pos_pas_map_loss + neg_pas_map_loss

        # return
        losses = {"student_loss": margin_mse_loss, "map_loss": map_loss}
        losses["loss"] = losses["student_loss"] + losses["map_loss"]
        return losses


class Dense2SparseMapSoftmaxSumCriterion(Dense2SparseMapSemanticIDFNormCriterion):
    def __init__(self, idf_path: str, map_method: str, norm_method: str):
        super().__init__(idf_path, map_method, norm_method)

        self.idf = self.mask_zero_2_neg_inf(self.idf)

    @staticmethod
    def mask_zero_2_neg_inf(x: Tensor) -> Tensor:
        x[x == 0] = -torch.inf
        # x[x == 0] = -10e30
        return x

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

        device = pred_pos.device

        # idf.shape = [vocab_size], teacher_***_pas_emb["encoded_embeddings"].shape = [vocab_size]
        pos_pas_map = self.map_builder(
            dense=pos_pas_emb["encoded_matrix"], sparse=teacher_pos_pas_emb["encoded_logits"]
        )
        pos_pas_semantic_idf = (1 / 2) * (
                self.norm_method(self.idf.to(device))
                + self.norm_method(self.mask_zero_2_neg_inf(teacher_pos_pas_emb["encoded_embeddings"]))
        )
        pos_pas_map_loss = self.kl_div(pos_pas_map, pos_pas_semantic_idf)

        neg_pas_map = self.map_builder(
            dense=neg_pas_emb["encoded_matrix"], sparse=teacher_neg_pas_emb["encoded_logits"]
        )
        neg_pas_semantic_idf = (1 / 2) * (
                self.norm_method(self.idf.to(device))
                + self.norm_method(self.mask_zero_2_neg_inf(teacher_neg_pas_emb["encoded_embeddings"]))
        )
        neg_pas_map_loss = self.kl_div(neg_pas_map, neg_pas_semantic_idf)

        # map_loss = query_map_loss + pos_pas_map_loss + neg_pas_map_loss
        map_loss = pos_pas_map_loss + neg_pas_map_loss

        # return
        losses = {"student_loss": margin_mse_loss, "map_loss": map_loss}
        losses["loss"] = losses["student_loss"] + losses["map_loss"]
        return losses


class SoftmaxIDF2MLMSumCriterion(Dense2SparseMapSemanticIDFNormCriterion):
    def __init__(self, idf_path: str, map_method: str, norm_method: str, target: str):
        super().__init__(idf_path, map_method, norm_method)

        self.idf = self.mask_zero_2_neg_inf(self.idf)
        self.targets = {
            "q": False,
            "pd": True,
            "nd": False,
        }
        if "q" in target:
            self.targets["q"] = True
        if "pd" in target:
            self.targets["pd"] = True
        if "nd" in target:
            self.targets["nd"] = True

    @staticmethod
    def mask_zero_2_neg_inf(x: Tensor) -> Tensor:
        x[x == 0] = -torch.inf
        # x[x == 0] = -10e30
        return x

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
        batch_idx = kwargs["batch_idx"]
        # gpl guided loss
        pred_pos, pred_neg = pos_score_res["relevance"], neg_score_res["relevance"]
        margin_mse_loss = self.margin_mse(pred_pos, pred_neg, label)

        device = pred_pos.device

        # idf.shape = [vocab_size], teacher_***_pas_emb["encoded_embeddings"].shape = [vocab_size]
        idf = self.norm_method(self.idf.to(device))

        dada_loss = torch.zeros((1,), device=device)
        # if batch_idx % 32 == 0:
        pos_vocab_avg = teacher_pos_pas_emb
        neg_vocab_avg = teacher_neg_pas_emb
        query_vocab_avg = teacher_query_emb

        pos_vocab_loss = self.kl_div(pos_vocab_avg, idf) if self.targets["pd"] else 0
        neg_vocab_loss = self.kl_div(neg_vocab_avg, idf) if self.targets["nd"] else 0
        query_vocab_loss = self.kl_div(query_vocab_avg, idf) if self.targets["q"] else 0

        dada_loss = pos_vocab_loss + neg_vocab_loss + query_vocab_loss

        # return
        losses = {"student_loss": margin_mse_loss, "map_loss": dada_loss}

        losses["loss"] = losses["student_loss"] + losses["map_loss"]
        return losses


class MLMSoftmaxSumCriterion(Dense2SparseMapSoftmaxSumCriterion):
    def __init__(self, idf_path: str, map_method: str, norm_method: str):
        super().__init__(idf_path, map_method, norm_method)

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

        device = pred_pos.device

        # idf.shape = [vocab_size], teacher_***_pas_emb["encoded_embeddings"].shape = [vocab_size]
        pos_pas_map = pos_pas_emb["encoded_vocabs"]
        pos_pas_semantic_idf = (1 / 2) * (
                self.norm_method(self.idf.to(device))
                + self.norm_method(self.mask_zero_2_neg_inf(teacher_pos_pas_emb["encoded_embeddings"]))
        )
        pos_pas_map_loss = self.kl_div(pos_pas_map, pos_pas_semantic_idf)

        neg_pas_map = neg_pas_emb["encoded_vocabs"]
        neg_pas_semantic_idf = (1 / 2) * (
                self.norm_method(self.idf.to(device))
                + self.norm_method(self.mask_zero_2_neg_inf(teacher_neg_pas_emb["encoded_embeddings"]))
        )
        neg_pas_map_loss = self.kl_div(neg_pas_map, neg_pas_semantic_idf)

        # map_loss = query_map_loss + pos_pas_map_loss + neg_pas_map_loss
        map_loss = pos_pas_map_loss + neg_pas_map_loss

        # return
        losses = {"student_loss": margin_mse_loss, "map_loss": map_loss}
        losses["loss"] = losses["student_loss"] + losses["map_loss"]
        return losses


class MLMSoftmaxSumPQTFIDFCriterion(Dense2SparseMapSoftmaxSumCriterion):
    def __init__(self, idf_path: str, map_method: str, norm_method: str, pqtfidf_path: str):
        super().__init__(idf_path, map_method, norm_method)

        self.pqtfidf = {}
        with open(pqtfidf_path, "r") as fs:
            num_lines = sum(1 for _ in fs)

        with open(pqtfidf_path, "r") as fs:
            for f in tqdm(fs, total=num_lines):
                f = json.loads(f)
                did = f["_id"]
                tokens = f["token_ids"]
                scores = f["scores"]
                pqtfidf = torch.zeros_like(self.idf)
                for token, score in zip(tokens, scores):
                    pqtfidf[token] = score
                self.pqtfidf[did] = pqtfidf

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
        ids = kwargs["ids"]

        # gpl guided loss
        pred_pos, pred_neg = pos_score_res["relevance"], neg_score_res["relevance"]
        margin_mse_loss = self.margin_mse(pred_pos, pred_neg, label)

        device = pred_pos.device

        # idf.shape = [vocab_size], teacher_***_pas_emb["encoded_embeddings"].shape = [vocab_size]
        pos_pas_map = pos_pas_emb["encoded_vocabs"]
        pos_pqtfidf = torch.stack(
            [self.pqtfidf[_id[0]] for _id in ids]
        ).to(device)
        pos_pas_semantic_idf = (1 / 2) * (
                self.norm_method(self.idf.to(device))
                + self.norm_method(self.mask_zero_2_neg_inf(pos_pqtfidf))
        )
        pos_pas_map_loss = self.kl_div(pos_pas_map, pos_pas_semantic_idf)

        neg_pas_map = neg_pas_emb["encoded_vocabs"]
        neg_pqtfidf = torch.stack(
            [self.pqtfidf[_id[1]] for _id in ids]
        ).to(device)
        neg_pas_semantic_idf = (1 / 2) * (
                self.norm_method(self.idf.to(device))
                + self.norm_method(self.mask_zero_2_neg_inf(neg_pqtfidf))
        )
        neg_pas_map_loss = self.kl_div(neg_pas_map, neg_pas_semantic_idf)

        # map_loss = query_map_loss + pos_pas_map_loss + neg_pas_map_loss
        map_loss = pos_pas_map_loss + neg_pas_map_loss

        # return
        losses = {"student_loss": margin_mse_loss, "map_loss": map_loss}
        losses["loss"] = losses["student_loss"] + losses["map_loss"]
        return losses


class MLMSoftmaxSumClusterTFIDFCriterion(Dense2SparseMapSoftmaxSumCriterion):
    def __init__(self, idf_path: str, map_method: str, norm_method: str, cluster_tfidf_dir: str):
        super().__init__(idf_path, map_method, norm_method)

        did2cluster_file_name = f"{cluster_tfidf_dir}/did2cluster.csv"
        tf_idf_file_name = f"{cluster_tfidf_dir}/tf-idf.jsonl"

        self.did2cluster = {}
        with open(did2cluster_file_name, "r") as fs:
            for idx, f in enumerate(fs):
                if idx == 0:
                    continue
                f = f.strip().split(",")
                did = f[0]
                cluster = f[1]
                self.did2cluster[did] = int(cluster)

        n_clusters = len(set(self.did2cluster.values()))
        self.tfs = torch.zeros((n_clusters, len(self.idf)))
        self.idfs = torch.zeros((n_clusters, len(self.idf)))
        with open(tf_idf_file_name, "r") as fs:
            num_lines = sum(1 for _ in fs)

        with open(tf_idf_file_name, "r") as fs:
            for idx, f in tqdm(enumerate(fs), total=num_lines):
                f = json.loads(f)
                cluster_id = f["_id"]
                idf_raw = f["idf"]
                tf_raw = f["tf"]

                self.tfs[idx] = torch.tensor(tf_raw)
                self.idfs[idx] = torch.tensor(idf_raw)

        """
        self.tfs = {}
        self.idfs = {}
        with open(tf_idf_file_name, "r") as fs:
            num_lines = sum(1 for _ in fs)

        with open(tf_idf_file_name, "r") as fs:
            for f in tqdm(fs, total=num_lines):
                f = json.loads(f)
                cluster_id = f["_id"]
                idf_raw = f["idf"]
                tf_raw = f["tf"]

                idf = []
                idf_idxs = []
                for idf_idx, idf_val in enumerate(idf_raw):
                    if idf_val > 0:
                        idf.append(idf_val)
                        idf_idxs.append(idf_idx)

                tf = []
                tf_idxs = []
                for tf_idx, tf_val in enumerate(tf_raw):
                    if tf_val > 0:
                        tf.append(tf_val)
                        tf_idxs.append(tf_idx)

                self.tfs[cluster_id] = {
                    "tf": tf,
                    "tf_idxs": tf_idxs,
                }
                self.idfs[cluster_id] = {
                    "idf": idf,
                    "idf_idxs": idf_idxs,
                }
        """

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

        device = pred_pos.device

        ids = kwargs["ids"]
        ###########
        """
        pos_doc_cluster_ids = [self.did2cluster[_id[0]] for _id in ids]
        neg_doc_cluster_ids = [self.did2cluster[_id[1]] for _id in ids]

        pos_docs_tfs = torch.zeros((len(ids), len(self.idf)), device=device)
        pos_docs_idfs = torch.zeros((len(ids), len(self.idf)), device=device)
        for idx, _id in enumerate(pos_doc_cluster_ids):
            tf_idxs = self.tfs[_id]["tf_idxs"]
            tfs = self.tfs[_id]["tf"]

            for tf_idx, tf in zip(tf_idxs, tfs):
                pos_docs_tfs[idx][tf_idx] = tf

            idf_idxs = self.idfs[_id]["idf_idxs"]
            idfs = self.idfs[_id]["idf"]

            for idf_idx, idf in zip(idf_idxs, idfs):
                pos_docs_idfs[idx][idf_idx] = idf

        neg_docs_tfs = torch.zeros((len(ids), len(self.idf)), device=device)
        neg_docs_idfs = torch.zeros((len(ids), len(self.idf)), device=device)
        for idx, _id in enumerate(neg_doc_cluster_ids):
            tf_idxs = self.tfs[_id]["tf_idxs"]
            tfs = self.tfs[_id]["tf"]

            for tf_idx, tf in zip(tf_idxs, tfs):
                neg_docs_tfs[idx][tf_idx] = tf

            idf_idxs = self.idfs[_id]["idf_idxs"]
            idfs = self.idfs[_id]["idf"]

            for idf_idx, idf in zip(idf_idxs, idfs):
                neg_docs_idfs[idx][idf_idx] = idf
        """
        pos_docs_tfs = self.tfs[[self.did2cluster[_id[0]] for _id in ids]]
        pos_docs_idfs = self.idfs[[self.did2cluster[_id[0]] for _id in ids]]
        neg_docs_tfs = self.tfs[[self.did2cluster[_id[1]] for _id in ids]]
        neg_docs_idfs = self.idfs[[self.did2cluster[_id[1]] for _id in ids]]

        # idf.shape = [vocab_size], teacher_***_pas_emb["encoded_embeddings"].shape = [vocab_size]
        pos_pas_voc = pos_pas_emb["encoded_vocabs"]
        pos_pas_guide = self.mask_zero_2_neg_inf(
            pos_docs_tfs * pos_docs_idfs
        ).to(device)
        pos_pas_map_loss = self.kl_div(
            pos_pas_voc.log_softmax(dim=-1),
            pos_pas_guide.softmax(dim=-1)
        )

        neg_pas_map = neg_pas_emb["encoded_vocabs"]
        neg_pas_guide = self.mask_zero_2_neg_inf(
            neg_docs_tfs * neg_docs_idfs
        ).to(device)
        neg_pas_map_loss = self.kl_div(
            neg_pas_map.log_softmax(dim=-1),
            neg_pas_guide.softmax(dim=-1)
        )

        # map_loss = query_map_loss + pos_pas_map_loss + neg_pas_map_loss
        map_loss = pos_pas_map_loss + neg_pas_map_loss

        # return
        losses = {"student_loss": margin_mse_loss, "map_loss": map_loss}
        losses["loss"] = losses["student_loss"] + losses["map_loss"]
        return losses


class Dense2SparseMapSemanticIDFNormBothCriterion(Dense2SparseMapSemanticIDFNormCriterion):
    def __init__(self, idf_path: str, map_method: str, norm_method: str):
        super().__init__(idf_path, map_method, norm_method)

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

        device = pred_pos.device

        # idf.shape = [vocab_size], teacher_***_pas_emb["encoded_embeddings"].shape = [vocab_size]
        pos_pas_map = self.norm_method(
            self.map_builder(
                pos_pas_emb["encoded_matrix"], teacher_pos_pas_emb["encoded_logits"]
            )
        )
        pos_pas_semantic_idf = (
                self.norm_method(self.idf.to(device))
                * self.norm_method(teacher_pos_pas_emb["encoded_embeddings"])
        )
        pos_pas_map_loss = self.kl_div(pos_pas_map, pos_pas_semantic_idf)

        neg_pas_map = self.norm_method(
            self.map_builder(
                neg_pas_emb["encoded_matrix"], teacher_neg_pas_emb["encoded_logits"]
            )
        )
        neg_pas_semantic_idf = (
                self.norm_method(self.idf.to(device))
                * self.norm_method(teacher_neg_pas_emb["encoded_embeddings"])
        )
        neg_pas_map_loss = self.kl_div(neg_pas_map, neg_pas_semantic_idf)

        # map_loss = query_map_loss + pos_pas_map_loss + neg_pas_map_loss
        map_loss = pos_pas_map_loss + neg_pas_map_loss

        # return
        losses = {"student_loss": margin_mse_loss, "map_loss": map_loss}
        losses["loss"] = losses["student_loss"] + losses["map_loss"]
        return losses


class Dense2SparseMapSemanticIDFNormCriterionAblation(Dense2SparseMapSemanticIDFNormCriterion):
    def __init__(self, idf_path: str, map_method: str, norm_method: str, ablation: str):
        super().__init__(idf_path, map_method, norm_method)
        self.ablation = ablation

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

        device = pred_pos.device

        # idf.shape = [vocab_size], teacher_***_pas_emb["encoded_embeddings"].shape = [vocab_size]
        pos_pas_map = self.map_builder(
            pos_pas_emb["encoded_matrix"], teacher_pos_pas_emb["encoded_logits"]
        )
        if self.ablation == "gen":
            pos_pas_semantic_idf = (
                self.norm_method(self.idf.to(device))
            )
        elif self.ablation == "spec":
            pos_pas_semantic_idf = (
                self.norm_method(teacher_pos_pas_emb["encoded_embeddings"])
            )
        else:
            raise NotImplementedError
        pos_pas_map_loss = self.kl_div(pos_pas_map, pos_pas_semantic_idf)

        neg_pas_map = self.map_builder(
            neg_pas_emb["encoded_matrix"], teacher_neg_pas_emb["encoded_logits"]
        )
        if self.ablation == "gen":
            neg_pas_semantic_idf = (
                self.norm_method(self.idf.to(device))
            )
        elif self.ablation == "spec":
            neg_pas_semantic_idf = (
                self.norm_method(teacher_neg_pas_emb["encoded_embeddings"])
            )
        else:
            raise NotImplementedError
        neg_pas_map_loss = self.kl_div(neg_pas_map, neg_pas_semantic_idf)

        # map_loss = query_map_loss + pos_pas_map_loss + neg_pas_map_loss
        map_loss = pos_pas_map_loss + neg_pas_map_loss

        # return
        losses = {"student_loss": margin_mse_loss, "map_loss": map_loss}
        losses["loss"] = losses["student_loss"] + losses["map_loss"]
        return losses


class Dense2SparseMapSemanticIDFNormCriterionWithoutGPL(Dense2SparseMapSemanticIDFNormCriterion):
    def __init__(self, idf_path: str, map_method: str, norm_method: str):
        super().__init__(idf_path, map_method, norm_method)

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
        losses = super().forward(
            query_emb, pos_pas_emb, neg_pas_emb,
            teacher_query_emb, teacher_pos_pas_emb, teacher_neg_pas_emb,
            pos_score_res, neg_score_res,
            label, *args, **kwargs
        )
        losses.pop("student_loss")
        losses["loss"] = losses["map_loss"]
        return losses


class ATDSCriterion(nn.Module):
    def __init__(self, idf_path: str, map_method: str):
        super().__init__()
        self.scales = {"margin_mse": 1}
        self.margin_mse = MarginMSELoss(scale=self.scales["margin_mse"])
        self.mse = MSELoss()

        # idf
        self.idf = tensor(json.load(open(idf_path, "r")))
        mean, std = torch.mean(self.idf), torch.std(self.idf)
        self.idf_norm = (self.idf - mean) / std

        # utils
        self.map_method = map_method

    def map_builder(self, dense: Tensor, sparse: Tensor) -> Tensor:
        output = einsum(
            dense, sparse,
            "batch len dense, batch len sparse -> batch dense sparse"
        )
        if self.map_method == "max":
            output = output.amax(dim=1)
        elif self.map_method == "mean":
            output = output.mean(dim=1)
        elif self.map_method == "sum":
            output = output.sum(dim=1)
        else:
            raise NotImplementedError
        return output

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
            batch_idx: int,
    ):
        # gpl guided loss
        pred_pos, pred_neg = pos_score_res["relevance"], neg_score_res["relevance"]
        margin_mse_loss = self.margin_mse(pred_pos, pred_neg, label)

        # ATDS
        # 0    ~ 1999 = idf
        # 2000 ~ 3999 = splade
        mode = "idf" if (batch_idx // 2000) % 2 == 1 else "splade"

        # min-max scaling
        # https://stackoverflow.com/questions/62178888/can-someone-explain-to-me-how-minmaxscaler-works

        atds_loss = 0
        if mode == "idf":
            # (batch, doc_len, 1)
            # (V) => (batch, doc_len) => (batch, doc_len, 1)
            idf = self.idf.to(pos_pas_emb["weight"].device)
            idf_max = idf.max()
            idf_min = idf.min()
            scaled_idf = (idf - idf_min) / (idf_max - idf_min)

            pos_emb_loss = self.mse(
                rearrange(pos_pas_emb["weight"], "batch doc_len l -> batch (doc_len l)"),
                scaled_idf[pos_pas_emb["input_ids"]]
            )
            neg_emb_loss = self.mse(
                rearrange(neg_pas_emb["weight"], "batch doc_len l -> batch (doc_len l)"),
                scaled_idf[neg_pas_emb["input_ids"]]
            )
            atds_loss = pos_emb_loss + neg_emb_loss

        elif mode == "splade":
            # (batch, doc_len, 1)
            # (batch, V) => (batch, doc_len) => (batch, doc_len, 1)
            atds_loss = 0

            teacher = torch.cat([teacher_pos_pas_emb["encoded_embeddings"], teacher_neg_pas_emb["encoded_embeddings"]])
            teacher_max = teacher.max()
            teacher_min = teacher.min()
            scaled_teacher = (teacher - teacher_min) / (teacher_max - teacher_min)

            for student_id, student_val, teacher_vals in zip(
                    torch.cat([pos_pas_emb["input_ids"], neg_pas_emb["input_ids"]]),
                    torch.cat(
                        [
                            rearrange(pos_pas_emb["weight"], "batch doc_len l -> batch (doc_len l)"),
                            rearrange(neg_pas_emb["weight"], "batch doc_len l -> batch (doc_len l)"),
                        ]
                    ),
                    scaled_teacher
            ):
                atds_loss += self.mse(
                    student_val,
                    teacher_vals[student_id]
                )

        # return
        losses = {"student_loss": margin_mse_loss, "atds_loss": atds_loss}
        losses["loss"] = losses["student_loss"] + losses["atds_loss"]
        return losses
