import logging
from itertools import chain
from typing import Dict, Union, Optional, Any, List, Callable

import beir
import torch
from beir.retrieval.evaluation import EvaluateRetrieval
from einops import rearrange
from lightning_utilities import apply_to_collection
from lightning_utilities.core.rank_zero import rank_zero_only
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.retrieval import RetrievalNormalizedDCG, RetrievalMAP, RetrievalMRR, RetrievalRecall, \
    RetrievalPrecision
from torchmetrics.utilities.data import dim_zero_cat, _flatten
from torchmetrics.utilities.distributed import gather_all_tensors

"""
class EvaluationMetric(Metric):
    full_state_update = False
    # dist_sync_on_step = True

    def __init__(self, k_values=None):
        super().__init__()
        if k_values is None:
            k_values = [1, 3, 5, 10, 20, 100]

        self.k_values = k_values
        configs = {
            "sync_on_compute": True,
            # "dist_sync_on_step": True,
            # "device": self.device,
        }
        self.ndcg = {
            f"NDCG@{k_value}": RetrievalNormalizedDCG(top_k=k_value, **configs)
            for k_value in self.k_values
        }
        self._map = {
            f"MAP@{k_value}": RetrievalMAP(top_k=k_value, **configs)
            for k_value in self.k_values
        }
        self.mrr = {
            f"MRR@{k_value}": RetrievalMRR(top_k=k_value, **configs)
            for k_value in self.k_values
        }
        self.precision = {
            f"Precision@{k_value}": RetrievalPrecision(top_k=k_value, **configs)
            for k_value in self.k_values
        }
        self.recall = {
            f"Recall@{k_value}": RetrievalRecall(top_k=k_value, **configs)
            for k_value in self.k_values
        }
        self.metrics = {
            "fl.ndcg": self.ndcg,
            "fl.map": self._map,
            "fl.mrr": self.mrr,
            "fl.precision": self.precision,
            "fl.recall": self.recall,
        }

    def update(self, index: Union[Tensor, int], pred: Tensor, label: Tensor, rerank: Union[Tensor, bool]):
        qids = index[:, 0]

        for overall_metric_obj in self.metrics.values():
            for metric_obj in overall_metric_obj.values():
                metric_obj.update(pred, label > 0, indexes=qids)

    def compute(self) -> Dict[str, Dict[str, float]]:

        result = {
            overall_metric_name: {
                metric_name: metric_obj.compute()
                for metric_name, metric_obj in overall_metric_obj.items()
            }
            for overall_metric_name, overall_metric_obj in self.metrics.items()
        }

        return result

"""
class EvaluationMetric(Metric):
    full_state_update = False

    def __init__(self, k_values=None):
        super().__init__()
        if k_values is None:
            k_values = [1, 3, 5, 10, 20, 100]

        self.k_values = k_values
        self.add_state("idxs", default=torch.empty(0), dist_reduce_fx="cat")
        self.add_state("labels", default=torch.empty(0), dist_reduce_fx="cat")
        self.add_state("preds", default=torch.empty(0), dist_reduce_fx="cat")
        self.add_state("reranks", default=torch.empty(0), dist_reduce_fx="cat")

    def _sync_dist(self, dist_sync_fn: Callable = gather_all_tensors, process_group: Optional[Any] = None) -> None:
        input_dict = {attr: getattr(self, attr) for attr in self._reductions}

        print("Syncing tensors")
        for attr, reduction_fn in self._reductions.items():
            # pre-concatenate metric states that are lists to reduce number of all_gather operations
            if reduction_fn == dim_zero_cat and isinstance(input_dict[attr], list) and len(input_dict[attr]) > 1:
                input_dict[attr] = [dim_zero_cat(input_dict[attr])]

        print("Gathering tensors")
        output_dict = apply_to_collection(
            input_dict,
            Tensor,
            dist_sync_fn,
            group=process_group or self.process_group,
        )

        self.number_of_devices = len(output_dict["idxs"])

        print("Reducing tensors")
        for attr, reduction_fn in self._reductions.items():
            # pre-processing ops (stack or flatten for inputs)

            if isinstance(output_dict[attr], list) and len(output_dict[attr]) == 0:
                setattr(self, attr, [])
                continue

            if isinstance(output_dict[attr][0], Tensor):
                output_dict[attr] = torch.cat(output_dict[attr])
            elif isinstance(output_dict[attr][0], list):
                output_dict[attr] = _flatten(output_dict[attr])

            if not (callable(reduction_fn) or reduction_fn is None):
                raise TypeError("reduction_fn must be callable or None")
            reduced = reduction_fn(output_dict[attr]) if reduction_fn is not None else output_dict[attr]
            setattr(self, attr, reduced)

    def update(self, index: Union[Tensor, int], pred: Tensor, label: Tensor, rerank: Union[Tensor, bool]):

        if type(index) == int:
            index = torch.ones_like(pred, device=self.device) * index
            self.idxs = torch.cat((self.idxs, index), dim=0)
        else:
            self.idxs = torch.cat((self.idxs, index), dim=0) if self.idxs.shape[0] != 0 else index
        self.labels = (
            torch.cat((self.labels, label), dim=0) if self.labels.shape[0] != 0 else label
        )
        self.preds = torch.cat((self.preds, pred), dim=0) if self.preds.shape[0] != 0 else pred
        if type(rerank) == bool:
            if rerank:
                rerank = torch.ones_like(pred)
            else:
                rerank = torch.zeros_like(pred)
            self.reranks = (
                torch.cat((self.reranks, rerank), dim=0) if self.reranks.shape[0] != 0 else rerank
            )
        else:
            self.reranks = (
                torch.cat((self.reranks, rerank), dim=0) if self.reranks.shape[0] != 0 else rerank
            )

    def compute(self) -> Dict[str, Dict[str, float]]:
        full_results = {}
        # reranks: Dict[str, Dict[str, any]] = {}
        qrels = {}

        if len(self.idxs.shape) == 3:
            self.idxs = rearrange(self.idxs, "gpus batch id_pair -> (gpus batch) id_pair")
            self.preds = rearrange(self.preds, "gpus batch -> (gpus batch)")
            self.labels = rearrange(self.labels, "gpus batch -> (gpus batch)")
            self.reranks = rearrange(self.reranks, "gpus batch -> (gpus batch)")

        print("Moving tensors to cpu as List")
        self.idxs = self.idxs.tolist()
        self.preds = self.preds.tolist()
        self.labels = self.labels.tolist()
        self.reranks = self.reranks.tolist()

        print(f"Computing metrics for {len(self.idxs)} samples")
        for (i, idx), pred, rerank, label in (
                zip(enumerate(self.idxs), self.preds, self.reranks, self.labels)
        ):
            qid, pid = idx
            qid = str(qid)
            pid = str(pid)

            if qid not in full_results:
                full_results[qid] = {}

            # if qid not in reranks:
            #     reranks[qid] = {}

            if qid not in qrels:
                qrels[qid] = {}

            full_results[qid][pid] = pred
            # reranks[qid][pid] = rerank.item()
            qrels[qid][pid] = 1 if label > 0 else 0
        print("Metrics building done")

        # re_rank_targets = list(
        #     filter(
        #         lambda item: item is not None,
        #         chain.from_iterable(
        #             map(
        #                 lambda q: map(
        #                     lambda p: (q, p) if reranks[q][p] == 1 else None,
        #                     reranks[q].keys(),
        #                 ),
        #                 reranks.keys(),
        #             )
        #         ),
        #     )
        # )

        # re_qrels = {}
        # re_results = {}
        # for target_qid, target_pid in re_rank_targets:
        #     if target_qid not in re_results:
        #         re_results[target_qid] = {}
        #         re_qrels[target_qid] = {}
        #
        #     re_results[target_qid][target_pid] = full_results[target_qid][target_pid]
        #     re_qrels[target_qid][target_pid] = qrels[target_qid][target_pid]

        full_ndcg, full_map, full_recall, full_precision = EvaluateRetrieval.evaluate(
            qrels, full_results, k_values=self.k_values,
            ignore_identical_ids=False,
        )
        full_mrr = EvaluateRetrieval.evaluate_custom(
            qrels, full_results, k_values=self.k_values, metric="mrr",
        )
        # re_ndcg, re_map, re_recall, re_precision, re_mrr = -1, -1, -1, -1, -1
        # if re_results:
        #     re_ndcg, re_map, re_recall, re_precision = EvaluateRetrieval.evaluate(
        #         re_qrels, re_results, k_values=self.k_values,
        #         ignore_identical_ids=False,
        #     )
        #     re_mrr = EvaluateRetrieval.evaluate_custom(
        #         re_qrels, re_results, k_values=self.k_values, metric="mrr"
        #     )
        output = {
            # full-rank
            "fl.ndcg": full_ndcg,
            "fl.map": full_map,
            "fl.recall": full_recall,
            "fl.precision": full_precision,
            "fl.mrr": full_mrr,
            # re-rank
            # "re.ndcg": re_ndcg,
            # "re.map": re_map,
            # "re.recall": re_recall,
            # "re.precision": re_precision,
            # "re.mrr": re_mrr,
        }
        return output
