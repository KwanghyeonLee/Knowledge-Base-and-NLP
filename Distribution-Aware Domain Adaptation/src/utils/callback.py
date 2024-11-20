import datetime
import os
import shutil
from collections import defaultdict
from itertools import product
from pathlib import Path
from time import sleep
from typing import Any, Optional, Sequence, Dict

import pytorch_lightning as pl
import torch
from beir.retrieval.evaluation import EvaluateRetrieval
from einops import einsum
from lightning_utilities.core.rank_zero import rank_zero_only
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import BasePredictionWriter
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor
from tqdm.rich import tqdm

from src.utils import get_pylogger

logger = get_pylogger(__name__)


class PredictionWriter(BasePredictionWriter):
    def write_on_epoch_end(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def __init__(self, output_dir):
        super().__init__(write_interval="batch")
        self.output_dir = Path(output_dir) / "predictions"
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def write_on_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        prediction: Any,
        batch_indices: Optional[Sequence[int]],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        torch.save(prediction, self.output_dir / f"b{batch_idx}_r{trainer.global_rank}.pt")


class RetrievalEvaluationWithDiskWriter(Callback):
    def __init__(self,
                 output_dir,
                 experiment_name: str = None,
                 method_name: str = None, model_type: str = None, dataset_name: str = None):
        super().__init__()
        self.top_k = 1000
        self.progress_bar_idx = 0
        self.progress_bar_id = None

        self.metric_result = None

        self.method_name = method_name
        self.model_type = model_type
        self.dataset_name = dataset_name
        self.per_query_metrics = defaultdict(dict)

        self.results = defaultdict(dict)
        self._zero_results = defaultdict(dict)
        self.qrels = defaultdict(dict)
        self._zero_qrels = defaultdict(dict)
        self.k_values = [1, 10, 100]

        # path / datetime now
        self.output_dir = Path(output_dir) / experiment_name if experiment_name is not None else Path(output_dir)
        print(f"{self.output_dir=}")
        self._prepare_dir()

    @rank_zero_only
    def _prepare_dir(self):
        if self.output_dir.is_dir():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        (self.output_dir / "q").mkdir(exist_ok=True, parents=True)
        (self.output_dir / "d").mkdir(exist_ok=True, parents=True)
        (self.output_dir / "s").mkdir(exist_ok=True, parents=True)
        (self.output_dir / "r").mkdir(exist_ok=True, parents=True)
        (self.output_dir / "j").mkdir(exist_ok=True, parents=True)

    @staticmethod
    def write_on_disk_custom(
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            save_target: Any,
            save_dir: str,
            file_name: str
    ):
        output_path = Path(save_dir) / file_name
        output_path.parent.mkdir(exist_ok=True, parents=True)
        torch.save(save_target, output_path)

    def write_on_disk(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule",
        target: Any, dtype: str, batch_idx: int = None, dataloader_idx: int = None,
    ):
        output_path = self.output_dir
        if dtype == "q":
            output_path /= "q"
        elif dtype == "d":
            output_path /= "d"
        elif dtype == "s":
            output_path /= "s"
        elif dtype == "r":
            output_path /= "r"
        else:
            (output_path / dtype).mkdir(exist_ok=True, parents=True)
            output_path /= dtype

        filename = f"b{batch_idx}" if batch_idx is not None else ""
        filename += f"_d{dataloader_idx}" if dataloader_idx is not None else ""
        filename += f"_r{pl_module.device.index}" if pl_module.device.index is not None else ""
        filename += ".pt"
        torch.save(target, output_path / filename)

        if type(target) is dict:
            ks = list(target.keys())
            for k in ks:
                try:
                    target[k].detach()
                except:
                    pass
                del target[k]
        else:
            target.detach()
            del target
        # clear cache
        torch.cuda.empty_cache()

    @rank_zero_only
    def _clear_tmp_files(self):
        shutil.rmtree(self.output_dir / "q")
        shutil.rmtree(self.output_dir / "d")
        shutil.rmtree(self.output_dir / "s")
        shutil.rmtree(self.output_dir / "r")
        shutil.rmtree(self.output_dir / "j")


        (self.output_dir / "q").mkdir(exist_ok=True, parents=True)
        (self.output_dir / "d").mkdir(exist_ok=True, parents=True)
        (self.output_dir / "s").mkdir(exist_ok=True, parents=True)
        (self.output_dir / "r").mkdir(exist_ok=True, parents=True)
        (self.output_dir / "j").mkdir(exist_ok=True, parents=True)
        logger.info("Cleared tmp files")

    @rank_zero_only
    def init_pbar(self, trainer, desc: str, total_batches: int):
        self.progress_bar_idx = 0
        self.progress_bar_id = trainer.progress_bar_callback._add_task(
            total_batches=total_batches,
            description=desc,
        )
        trainer.progress_bar_callback.refresh()

    # @rank_zero_only
    def update_pbar(self, trainer):
        self.progress_bar_idx += 1
        trainer.progress_bar_callback._update(self.progress_bar_id, self.progress_bar_idx)
        trainer.progress_bar_callback.refresh()

    @rank_zero_only
    def close_pbar(self, trainer, pl_module):
        trainer.progress_bar_callback._update_metrics(trainer, pl_module)
        trainer.progress_bar_callback.progress.update(self.progress_bar_id, advance=0, visible=False)
        trainer.progress_bar_callback.refresh()
        self.progress_bar_idx = 0
        self.progress_bar_id = None

    # @rank_zero_only
    @torch.no_grad()
    def update_metrics(self, idxs, preds, labels):
        for (i, idx), pred, label in (
                zip(enumerate(idxs), preds, labels)
        ):
            qid, pid = idx
            qid = str(qid)
            pid = str(pid)

            # store only top 10 results
            if len(self.results[qid]) > self.top_k:
                current_scores = self.results[qid].values()

                if pred > min(current_scores):
                    del_did = min(self.results[qid], key=self.results[qid].get)
                    self.results[qid].pop(del_did)

                    self.results[qid][pid] = pred

            else:
                self.results[qid][pid] = pred

            # appending qrels
            if type(label) is int:
                if label > 0:
                    self.qrels[qid][pid] = label
                else:
                    self.qrels[qid][pid] = 0
            elif type(label) is bool:
                if label is True:
                    self.qrels[qid][pid] = 1
                else:
                    self.qrels[qid][pid] = 0
            else:
                raise ValueError(f"Unknown label type: {type(label)}")
    
    @rank_zero_only
    def sum_metrics(self, full_ndcg, full_map, full_recall, full_precision, full_mrr):
        for key in self.metric_result["fl.ndcg"]:
            self.metric_result["fl.ndcg"][key] += full_ndcg[key]
        for key in self.metric_result["fl.map"]:
            self.metric_result["fl.map"][key] += full_map[key]
        for key in self.metric_result["fl.recall"]:
            self.metric_result["fl.recall"][key] += full_recall[key]
        for key in self.metric_result["fl.precision"]:
            self.metric_result["fl.precision"][key] += full_precision[key]
        for key in self.metric_result["fl.mrr"]:
            self.metric_result["fl.mrr"][key] += full_mrr[key]

    @rank_zero_only
    def average_metrics(self, number_of_dataloaders: int):
        logger.info(f"averaging metrics ...")
        for key in self.metric_result["fl.ndcg"]:
            self.metric_result["fl.ndcg"][key] /= number_of_dataloaders
        for key in self.metric_result["fl.map"]:
            self.metric_result["fl.map"][key] /= number_of_dataloaders
        for key in self.metric_result["fl.recall"]:
            self.metric_result["fl.recall"][key] /= number_of_dataloaders
        for key in self.metric_result["fl.precision"]:
            self.metric_result["fl.precision"][key] /= number_of_dataloaders
        for key in self.metric_result["fl.mrr"]:
            self.metric_result["fl.mrr"][key] /= number_of_dataloaders

    @rank_zero_only
    def evaluate_metrics_per_query(self, dataloader):
        qidx2qid = dataloader.dataset.qidx2qid
        for qid in self._zero_qrels:
            full_ndcg, full_map, full_recall, full_precision = EvaluateRetrieval.evaluate(
                {qid: self._zero_qrels[qid]}, {qid: self._zero_results[qid]}, k_values=self.k_values,
                ignore_identical_ids=False,
            )
            full_mrr = EvaluateRetrieval.evaluate_custom(
                {qid: self._zero_qrels[qid]}, {qid: self._zero_results[qid]}, k_values=self.k_values, metric="mrr",
            )
            real_qid = qidx2qid[int(qid)]
            self.per_query_metrics[real_qid] = {
                "fl.ndcg": full_ndcg,
                "fl.map": full_map,
                "fl.recall": full_recall,
                "fl.precision": full_precision,
                "fl.mrr": full_mrr,
            }

    @rank_zero_only
    def evaluate_metrics(self):
        full_ndcg, full_map, full_recall, full_precision = EvaluateRetrieval.evaluate(
            self._zero_qrels, self._zero_results, k_values=self.k_values,
            ignore_identical_ids=False,
        )
        full_mrr = EvaluateRetrieval.evaluate_custom(
            self._zero_qrels, self._zero_results, k_values=self.k_values, metric="mrr",
        )

        if self.metric_result is None:
            self.metric_result = {
                "fl.ndcg": full_ndcg,
                "fl.map": full_map,
                "fl.recall": full_recall,
                "fl.precision": full_precision,
                "fl.mrr": full_mrr,
            }

        else:
            self.sum_metrics(full_ndcg, full_map, full_recall, full_precision, full_mrr)

    @rank_zero_only
    def print_metrics(self):
        for key in self.metric_result:
            if type(self.metric_result[key]) is not dict:
                continue
            logger.info(f"average {key}: {self.metric_result[key]}")

    @rank_zero_only
    def log_metrics(self, trainer, pl_module, dataloaders):
        self.average_metrics(len(dataloaders))
        for key in self.metric_result:
            if type(self.metric_result[key]) is not dict:
                continue
            pl_module.log_dict(self.metric_result[key])
        self.print_metrics()

    @rank_zero_only
    def log_per_query_metrics(self, trainer, pl_module):
        logger.info("logging per query metrics ...")
        self.write_on_disk_custom(
            trainer=trainer, pl_module=pl_module,
            save_target=self.per_query_metrics,
            save_dir=f"/workspace/research/perquery/{self.model_type}/{self.method_name}", file_name=f"{self.dataset_name}.pt"
        )
        logger.info("per query metrics logged")

    def clear_metrics(self):
        del self.metric_result
        self.metric_result = None

        del self.results
        del self._zero_results
        del self._zero_qrels
        del self.qrels
        self.results = defaultdict(dict)
        self._zero_results = defaultdict(dict)
        self.qrels = defaultdict(dict)
        self._zero_qrels = defaultdict(dict)

        torch.cuda.empty_cache()
        logger.info("cleared cache")

    def record_job_done(self, trainer, pl_module):
        torch.save(1, self.output_dir / "j" / f"r{pl_module.device.index}.pt")

    def check_job_done(self, trainer, pl_module):
        return all(
            map(
                lambda dev_id: (self.output_dir / "j" / f"r{dev_id}.pt").exists(),
                pl_module.trainer.device_ids
            )
        )

    @rank_zero_only
    def clear_job_done(self, trainer, pl_module):
        for dev_id in pl_module.trainer.device_ids:
            (self.output_dir / "j" / f"r{dev_id}.pt").unlink()

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
        **kwargs
    ) -> None:
        # save query
        # check whether the query is empty
        if outputs["save"]["qs"]["idxs"].shape[0] != 0:
            self.write_on_disk(trainer=trainer, pl_module=pl_module, target=outputs["save"]["qs"], dtype="q",
                               batch_idx=batch_idx, dataloader_idx=dataloader_idx)
        # save doc
        # check whether the doc is empty
        if outputs["save"]["ds"]["idxs"].shape[0] != 0:
            self.write_on_disk(trainer=trainer, pl_module=pl_module, target=outputs["save"]["ds"], dtype="d",
                               batch_idx=batch_idx, dataloader_idx=dataloader_idx)

    @torch.no_grad()
    def _score_q_d(self, trainer, pl_module, dataloader_idxs, dataloaders, device):
        for dataloader_idx in dataloader_idxs:
            dataloader = dataloaders[dataloader_idx]

            qrels = dataloader.dataset.qrels
            queries = list(
                filter(
                    lambda s: int(s.stem.split("_d")[-1].split("_")[0]) == dataloader_idx,
                    (self.output_dir / "q").glob("*.pt")
                )
            )
            docs = list(
                filter(
                    lambda s: int(s.stem.split("_d")[-1].split("_")[0]) == dataloader_idx,
                    (self.output_dir / "d").glob("*.pt")
                )
            )
            docs = list(filter(
                lambda i: int(i.stem.split("_r")[-1]) == device.index, docs
            ))

            self.init_pbar(trainer, f"Data {dataloader_idx}: Scoring QDs", len(queries) * len(docs))
            for (q_idx, query), (d_idx, doc) in product(enumerate(queries), enumerate(docs)):
                query = torch.load(query)
                q_ids = query["idxs"].to(device)
                q_embs = query["embs"].to(device)

                doc = torch.load(doc)
                d_ids = doc["idxs"].to(device)
                d_embs = doc["embs"].to(device)

                idxs = torch.cartesian_prod(q_ids, d_ids)
                scores = einsum(q_embs, d_embs, "query vocab, doc vocab -> query doc").flatten()
                labels = torch.stack(
                    [
                        qrels[qid, pid]
                        for qid, pid in idxs
                    ],
                ).to(device)

                self.write_on_disk(trainer=trainer, pl_module=pl_module, target={
                    "idxs": idxs,
                    "scores": scores,
                    "labels": labels,
                }, dtype="s", batch_idx=int(f"1{q_idx}0{d_idx}"), dataloader_idx=dataloader_idx)

                self.update_pbar(trainer)

                for var in [q_ids, q_embs, d_ids, d_embs, idxs, scores, labels, query, doc]:
                    if type(var) is dict:
                        ks = list(var.keys())
                        for k in ks:
                            var[k].detach()
                            del var[k]
                    elif type(var) is Tensor:
                        var.detach()
                        del var
                    else:
                        del var
                    torch.cuda.empty_cache()

            torch.cuda.empty_cache()

            self._wait_for_other_gpus(trainer, pl_module)
            self.close_pbar(trainer, pl_module)

    @torch.no_grad()
    def _gather_scores(self, trainer, pl_module, dataloader_idxs, dataloaders, device):
        for dataloader_idx in dataloader_idxs:
            score_files = list(
                filter(
                    lambda s: int(s.stem.split("_d")[-1].split("_")[0]) == dataloader_idx,
                    (self.output_dir / "s").glob("*.pt")
                )
            )
            score_files = list(filter(
                lambda i: int(i.stem.split("_r")[-1]) == device.index, score_files
            ))

            self.init_pbar(trainer, f"Data {dataloader_idx}: Gathering scores", len(score_files))
            for score_file in score_files:
                score = torch.load(score_file)
                idxs = score["idxs"].tolist()
                scores = score["scores"].tolist()
                labels = score["labels"].tolist()

                self.update_metrics(idxs, scores, labels)
                self.update_pbar(trainer)

                for var in [score, idxs, scores, labels]:
                    if type(var) is dict:
                        ks = list(var.keys())
                        for k in ks:
                            var[k].detach()
                            del var[k]
                    elif type(var) is Tensor:
                        var.detach()
                        del var
                    else:
                        del var
                    torch.cuda.empty_cache()

            self.write_on_disk(trainer=trainer, pl_module=pl_module, target={
                "results": self.results,
                "qrels": self.qrels,
            }, dtype="r", batch_idx=0, dataloader_idx=dataloader_idx)

            self._wait_for_other_gpus(trainer, pl_module)
            self.close_pbar(trainer, pl_module)

    @torch.no_grad()
    @rank_zero_only
    def _collect_scores(self, trainer, pl_module, dataloader_idxs, dataloaders, device):
        for dataloader_idx in dataloader_idxs:
            sub_score_files = list(
                filter(
                    lambda s: int(s.stem.split("_d")[-1].split("_")[0]) == dataloader_idx,
                    (self.output_dir / "r").glob("*.pt")
                )
            )

            self.init_pbar(trainer, f"Data {dataloader_idx}: Collecting scores", len(sub_score_files))
            for sub_score_file in sub_score_files:
                sub_score = torch.load(sub_score_file)
                sub_results = sub_score["results"]
                sub_qrels = sub_score["qrels"]

                for qid, result in sub_results.items():
                    for did, score in result.items():
                        self._zero_results[qid][did] = score

                for qid, qrel in sub_qrels.items():
                    for did, rel in qrel.items():
                        self._zero_qrels[qid][did] = rel

                self.update_pbar(trainer)

            # dataloader = dataloaders[dataloader_idx]
            # qidx2qid = dataloader.dataset.qidx2qid
            # pidx2pid = dataloader.dataset.pidx2pid

            # converted_results = defaultdict(dict)
            # for qid in self._zero_results:
            #     for pid in self._zero_results[qid]:
            #         converted_results[str(qidx2qid[int(qid)])][str(pidx2pid[int(pid)])] = float(self._zero_results[qid][pid])
            #
            # self.write_on_disk(
            #     trainer=trainer, pl_module=pl_module, target={
            #         "results": converted_results,
            #     },
            #     dtype="results",
            #     batch_idx=0,
            #     dataloader_idx=0
            # )
            self.close_pbar(trainer, pl_module)

    def _wait_for_other_gpus(self, trainer, pl_module):
        sleep(1)
        # all gpus should wait for rank 0 to finish
        self.record_job_done(trainer, pl_module)
        while not self.check_job_done(trainer, pl_module):
            # wait for other devices to finish
            pass
        sleep(1)
        self.clear_job_done(trainer, pl_module)

    def _on_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule",
                      dataloaders: list) -> None:
        device = pl_module.device
        dataloader_idxs = [i for i in range(len(dataloaders))]

        # write to s
        self._score_q_d(trainer, pl_module, dataloader_idxs, dataloaders, device)
        # write to r
        self._gather_scores(trainer, pl_module, dataloader_idxs, dataloaders, device)

        # ============ BUFFER: WAITING FOR OTHER GPUS ============ #
        self._wait_for_other_gpus(trainer, pl_module)
        # ======================================================= #

        # collect r
        self._collect_scores(trainer, pl_module, dataloader_idxs, dataloaders, device)
        # evaluate
        self.evaluate_metrics()
        self.evaluate_metrics_per_query(dataloader=dataloaders[0])

        self.log_metrics(trainer, pl_module, dataloaders)
        self.log_per_query_metrics(trainer, pl_module)
        # ============ BUFFER: WAITING FOR OTHER GPUS ============ #
        self._wait_for_other_gpus(trainer, pl_module)
        # ======================================================= #

        self._clear_tmp_files()
        self.clear_metrics()

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._on_epoch_end(trainer, pl_module, trainer.val_dataloaders)

    def on_test_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
        **kwargs
    ) -> None:
        self.on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def on_test_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._on_epoch_end(trainer, pl_module, trainer.test_dataloaders)

    def on_predict_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
        **kwargs
    ) -> None:
        self.on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)


if __name__ == '__main__':
    from tqdm import tqdm as raw_tqdm
    writer = RetrievalEvaluationWithDiskWriter(output_dir="/workspace/tmp/bioasq")
    print("writer initialized")

    score_files = list(
        filter(
            lambda s: int(s.stem.split("_d")[-1].split("_")[0]) == 0,
            (writer.output_dir / "s").glob("*.pt")
        )
    )

    print("scores files: ", len(score_files))

    for score_file in raw_tqdm(score_files):
        score = torch.load(score_file)
        idxs = score["idxs"].tolist()
        scores = score["scores"].tolist()
        labels = score["labels"].tolist()

        writer.update_metrics(idxs, scores, labels)
    print("metrics updated")
    writer.evaluate_metrics()
    print("metrics evaluated")
    for key in writer.metric_result:
        if type(writer.metric_result[key]) is not dict:
            continue
        print(f"average {key}: {writer.metric_result[key]}")
    print("done")