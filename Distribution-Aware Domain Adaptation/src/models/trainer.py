import copy
import deepspeed
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import wandb
from einops import einsum
from lightning_utilities.core.rank_zero import rank_zero_only
from pytorch_lightning.utilities.types import (
    EVAL_DATALOADERS,
    STEP_OUTPUT,
    TRAIN_DATALOADERS,
)
from torch import Tensor, nn, tensor
from torch.optim import Optimizer, lr_scheduler
from torchmetrics import MeanMetric
from torchmetrics.retrieval import RetrievalNormalizedDCG
from transformers import DistilBertTokenizer, optimization

from src.losses.hybrid import Dense2SparseMapCriterion, ATDSCriterion
from src.metrics.eval import EvaluationMetric
from src.models.components import RetrieverBase


class RetrievalModel(pl.LightningModule):
    """if LightningDataModule is injected to trainer, following methods just need pass.

    if not, following methods must be implemented. However, this project uses LightningDataModule.
    So, following dataloader methods became just boilerplate of LightningModule.
    """

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        pass

    def test_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def val_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def __init__(
            self,
            model: RetrieverBase,
            train_loss: nn.Module,
            val_loss: EvaluationMetric,
            optimizer: Optimizer,
            scheduler: lr_scheduler,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        # teacher is not needed for inference and evaluation.
        self.save_hyperparameters(logger=False, ignore=["train_loss", "val_loss"])

        self.model = model

        self.train_criterion = train_loss
        self.eval_criterion: EvaluationMetric = val_loss
        self.current_test_data_idx = 0
        self.current_val_data_idx = 0
        self.val_criterion: EvaluationMetric = copy.deepcopy(val_loss)
        self.test_criterion: EvaluationMetric = copy.deepcopy(val_loss)

        # self.idf = tensor(json.load(open(f"/workspace/gpl/{self.trainer.datamodule.test_datasets[0]}/idf/idfs_zero_unseen.json", "r")))

        # create the queue
        # self.register_buffer("pos_idf_criterion", torch.zeros(size=(self.model.mlm.predictions.bias.shape[0], )))
        # self.register_buffer("neg_idf_criterion", torch.zeros(size=(self.model.mlm.predictions.bias.shape[0], )))
        # self.register_buffer("query_idf_criterion", torch.zeros(size=(self.model.mlm.predictions.bias.shape[0], )))

    # hyper parameters
    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        if issubclass(self.hparams.optimizer.func, deepspeed.ops.adam.cpu_adam.DeepSpeedCPUAdam):
            optimizer = self.hparams.optimizer(model_params=self.parameters())
        else:
            optimizer = self.hparams.optimizer(params=self.parameters())
        scheduler = self.hparams.scheduler(optimizer=optimizer)

        return {
            "optimizer": optimizer,
            "interval": "step",
            "frequency": 1,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # or 'epoch'
                # "monitor": "loss",
                "frequency": 1,
            },
        }

    def teardown(self, stage: str) -> None:
        print()

    # For validation, test, prediction modes some model's need to change its internal setting (eg. peq-colbert)
    # this method calls `{student model}.set_validation()`

    def _model_prep(self) -> None:
        self.model.set_validation()

    def on_train_start(self) -> None:
        pass
        # self.pos_idf_criterion = self.pos_idf_criterion.to(self.device)
        # self.neg_idf_criterion = self.neg_idf_criterion.to(self.device)
        # self.query_idf_criterion = self.query_idf_criterion.to(self.device)

    def on_validation_start(self) -> None:
        self._model_prep()

        if type(self.trainer.val_dataloaders) == list:
            if type(self.val_criterion) == nn.ModuleList:
                return
            self.val_criterion = nn.ModuleList([
                copy.deepcopy(self.eval_criterion)
                for _ in range(len(self.trainer.val_dataloaders))
            ])

    def on_test_start(self) -> None:
        self._model_prep()

        if type(self.trainer.test_dataloaders) == list:
            if type(self.test_criterion) == nn.ModuleList:
                return
            self.test_criterion = nn.ModuleList([
                copy.deepcopy(self.eval_criterion)
                for _ in range(len(self.trainer.test_dataloaders))
            ])

    def on_predict_start(self) -> None:
        self._model_prep()

    # @@@@@@@@@@@@@@@@@ TRAINING methods @@@@@@@@@@@@@@@@@ #
    def training_step(self, batch: Tuple[Tensor, Tensor, Tensor, Tensor], batch_idx: int):
        """This method defines how model is trained without DISTILLATION.

        :param batch: (features, labels, idx)
            features: query_feature, (pos_pas_feature, neg_pas_feature)
            labels: answer of dataset. (margin between q-pos_doc & q-neg_doc)
            idx: this is order of dataset that each model gets. (this can be same as batch_idx but mostly different)
        :param batch_idx: index from dataset batch
        :return:
        """
        _, ids, queries, docs, labels = batch
        pos_docs, neg_docs = docs

        train_queries_emb, train_pos_docs_emb, train_neg_docs_emb = self.model.encode(
            queries, pos_docs, neg_docs
        )

        train_pos_score = self.model.score(train_queries_emb, train_pos_docs_emb)
        train_neg_score = self.model.score(train_queries_emb, train_neg_docs_emb)

        # vocab.shape = [batch_size, vocab_size] => [batch_size, vocab_probs]
        # query_vocab = train_queries_emb["encoded_vocabs"]
        # pos_vocab = train_pos_docs_emb["encoded_vocabs"]
        # neg_vocab = train_neg_docs_emb["encoded_vocabs"]

        # vocab_avg.shape = [vocab_size]
        # query_vocab_avg = query_vocab.mean(dim=0)
        # pos_vocab_avg = pos_vocab.mean(dim=0)
        # neg_vocab_avg = neg_vocab.mean(dim=0)

        # self.query_idf_criterion += query_vocab_avg
        # self.pos_idf_criterion += pos_vocab_avg
        # self.neg_idf_criterion += neg_vocab_avg

        # self.query_idf_criterion = self.all_gather(query_vocab_avg).mean(dim=0)
        # self.pos_idf_criterion = self.all_gather(pos_vocab_avg).mean(dim=0)
        # self.neg_idf_criterion = self.all_gather(neg_vocab_avg).mean(dim=0)

        losses: Dict[str, Tensor] = self.train_criterion(
            query_emb=train_queries_emb,
            pos_pas_emb=train_pos_docs_emb,
            neg_pas_emb=train_neg_docs_emb,
            # teacher_query_emb=self.query_idf_criterion,
            # teacher_pos_pas_emb=self.pos_idf_criterion,
            # teacher_neg_pas_emb=self.neg_idf_criterion,
            teacher_query_emb=None,
            teacher_pos_pas_emb=None,
            teacher_neg_pas_emb=None,
            pos_score_res=train_pos_score,
            neg_score_res=train_neg_score,
            label=labels,
            batch_idx=batch_idx,
            ids=ids
        )

        # if batch_idx % 32 == 0:
        #     # clear the queue
        #     self.query_idf_criterion = self.query_idf_criterion * 0
        #     self.pos_idf_criterion = self.pos_idf_criterion * 0
        #     self.neg_idf_criterion = self.neg_idf_criterion * 0

        return losses

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        """This method collects output if ddp is enabled.

        :param output: this output is from training_step
        :return:
        """
        gathered_output = dict(map(lambda i: (i[0], i[1].mean()), outputs.items()))
        self.log_dict(gathered_output)

    # def backward(self, loss: Tensor, *args: Any, **kwargs: Any) -> None:
    #     super().backward(loss, *args, **kwargs)
    #     print()

    # @@@@@@@@@@@@@@@@@ VALIDATION methods @@@@@@@@@@@@@@@@@ #
    @torch.no_grad()
    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        """This method is used for testing.

        :param batch: (features, labels, idx)
            features: (query, positive passage, negative passage)
            labels: answer (score of query-passage, from BEIR)
            idx: index of query-passage pair
        :param batch_idx: index of batch
        :param dataloader_idx: index of dataloader
        :return: {
            "loss": loss,
            "idxs": idxs,
            "relevance": relevance,
            "labels": labels,
            "dataloader_idx": dataloader_idx,
        }
        """
        ids, strings = batch
        is_query = ids[:, 1] == -1
        is_doc = ids[:, 0] == -1

        q_toks = {
            key: strings[key][is_query]
            for key in strings
        }
        d_toks = {
            key: strings[key][is_doc]
            for key in strings
        }
        q_embs = {"encoded_embeddings": []}
        if q_toks["input_ids"].shape[0] > 0:
            q_embs = self.model.encode_query(q_toks)

        d_embs = {"encoded_embeddings": []}
        if d_toks["input_ids"].shape[0] > 0:
            d_embs = self.model.encode_passage(d_toks)
        # emb 여러개 나옴.
        qs = {
            "idxs": ids[is_query][:, 0],
            "embs": q_embs["encoded_embeddings"],
        }
        ds = {
            "idxs": ids[is_doc][:, 1],
            "embs": d_embs["encoded_embeddings"],
        }

        return {
            "save": {
                "qs": qs,
                "ds": ds,
            },
        }

    # @@@@@@@@@@@@@@@@@ TEST methods @@@@@@@@@@@@@@@@@ #
    # following methods exactly do same logic as validation
    def test_step(self, batch: Any, batch_idx: int, **kwargs):
        return self.validation_step(batch, batch_idx, **kwargs)
        # output = self.validation_step(batch, batch_idx)
        # if type(output["save"]["qs"]["embs"]) is Tensor:
        #     output["save"]["qs"]["vocab"] = self.model.mlm(output["save"]["qs"]["embs"])
        # if type(output["save"]["ds"]["embs"]) is Tensor:
        #     output["save"]["ds"]["vocab"] = self.model.mlm(output["save"]["ds"]["embs"])
        # return output


class GPLRetrievalModel(RetrievalModel):
    def __init__(self, model: RetrieverBase,
                 train_loss: nn.Module, val_loss: EvaluationMetric, optimizer: Optimizer,
                 scheduler: lr_scheduler,
                 ):
        super().__init__(model, train_loss, val_loss, optimizer, scheduler)
        self.dense = model

        self.save_hyperparameters(logger=False, ignore=["train_loss", "val_loss"])

    def on_validation_start(self) -> None:
        self.dense.train(False)
        self.dense.eval()

        super().on_validation_start()

    def on_train_start(self) -> None:
        self.dense.train(True)

        super().on_test_start()

    def training_step(self, batch: Any, batch_idx: int) -> Dict[str, Tensor]:
        """

        Args:
            batch:
            batch_idx:
            optimizer_idx:

        Returns:

        """

        _, _, queries, docs, labels = batch
        dense_queries = queries
        pos_docs, neg_docs = docs

        train_queries_emb, train_pos_docs_emb, train_neg_docs_emb = self.dense.encode(
            queries, pos_docs, neg_docs
        )

        train_pos_score = self.dense.score(train_queries_emb, train_pos_docs_emb)
        train_neg_score = self.dense.score(train_queries_emb, train_neg_docs_emb)

        losses: Dict[str, Tensor] = self.train_criterion(
            query_emb=None,
            pos_pas_emb=None,
            neg_pas_emb=None,
            teacher_query_emb=None,
            teacher_pos_pas_emb=None,
            teacher_neg_pas_emb=None,
            pos_score_res=train_pos_score,
            neg_score_res=train_neg_score,
            label=labels,
            batch_idx=batch_idx,
        )
        return losses

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        """This method collects output if ddp is enabled.

        :param output: this output is from training_step
        :return:
        """
        gathered_output = dict(map(lambda i: (i[0], i[1].mean()), outputs.items()))
        self.log_dict(gathered_output)

    @torch.no_grad()
    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        """This method is used for testing.

        :param batch: (features, labels, idx)
            features: (query, positive passage, negative passage)
            labels: answer (score of query-passage, from BEIR)
            idx: index of query-passage pair
        :param batch_idx: index of batch
        :param dataloader_idx: index of dataloader
        :return: {
            "loss": loss,
            "idxs": idxs,
            "relevance": relevance,
            "labels": labels,
            "dataloader_idx": dataloader_idx,
        }
        """
        ids, strings = batch
        emb = self.dense.encode_passage(strings)
        is_query = ids[:, 1] == -1
        is_doc = ids[:, 0] == -1
        # emb 여러개 나옴.
        qs = {
            "idxs": ids[is_query][:, 0],
            "embs": emb["encoded_embeddings"][is_query],
        }
        ds = {
            "idxs": ids[is_doc][:, 1],
            "embs": emb["encoded_embeddings"][is_doc],
        }

        return {
            "save": {
                "qs": qs,
                "ds": ds,
            },
        }

    def test_step(self, batch: Any, batch_idx: int, **kwargs):
        return self.validation_step(batch, batch_idx, **kwargs)

    def on_test_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int = 0, **kwargs) -> None:
        super().on_test_batch_start(batch, batch_idx, dataloader_idx=0, **kwargs)


class HybridRetrievalModel(RetrievalModel):
    def __init__(self, model: RetrieverBase,
                 train_loss: nn.Module, val_loss: EvaluationMetric, optimizer: Optimizer,
                 scheduler: lr_scheduler,
                 sparse: RetrieverBase = None,
                 ):
        super().__init__(model, train_loss, val_loss, optimizer, scheduler)
        self.dense = model
        self.sparse = sparse

        if self.sparse:
            self.sparse.requires_grad_(False)

        self.save_hyperparameters(logger=False, ignore=["train_loss", "val_loss", "sparse"])

        # self.emb_voc_map_avg = torch.zeros((768, 30522))
        # self.emb_voc_map_square_avg = torch.zeros((768, 30522))
        # self.emb_voc_map_cnt = 0
        # self.cnt = 0

    def on_validation_start(self) -> None:
        self.dense.train(False)
        self.dense.eval()

        if self.sparse:
            self.sparse.training = False
            self.sparse.eval()
            self.sparse.train(False)

        super().on_validation_start()

        # self.emb_voc_map_avg = self.emb_voc_map_avg.to(self.device)
        # self.emb_voc_map_square_avg = self.emb_voc_map_square_avg.to(self.device)

    def on_train_start(self) -> None:
        self.dense.train(True)

        if self.sparse:
            self.sparse.training = False
            self.sparse.eval()
            self.sparse.train(False)

        super().on_test_start()

    def training_step(self, batch: Any, batch_idx: int) -> Dict[str, Tensor]:
        """

        Args:
            batch:
            batch_idx:
            optimizer_idx:

        Returns:

        """

        _, ids, queries, docs, labels = batch
        dense_queries, sparse_queries = queries
        (dense_pos_docs, dense_neg_docs), (sparse_pos_docs, sparse_neg_docs) = docs

        train_queries_emb, train_pos_docs_emb, train_neg_docs_emb = self.dense.encode(
            dense_queries, dense_pos_docs, dense_neg_docs
        )

        train_pos_score = self.dense.score(train_queries_emb, train_pos_docs_emb)
        train_neg_score = self.dense.score(train_queries_emb, train_neg_docs_emb)

        with torch.no_grad():
            teach_queries_emb, teach_pos_docs_emb, teach_neg_docs_emb = self.sparse.encode(
                sparse_queries, sparse_pos_docs, sparse_neg_docs
            )

        self.train_criterion: Union[Dense2SparseMapCriterion, ATDSCriterion]
        losses: Dict[str, Tensor] = self.train_criterion(
            query_emb=train_queries_emb,
            pos_pas_emb=train_pos_docs_emb,
            neg_pas_emb=train_neg_docs_emb,
            teacher_query_emb=teach_queries_emb,
            teacher_pos_pas_emb=teach_pos_docs_emb,
            teacher_neg_pas_emb=teach_neg_docs_emb,
            pos_score_res=train_pos_score,
            neg_score_res=train_neg_score,
            label=labels,
            batch_idx=batch_idx,
            ids=ids
        )

        return losses

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        """This method collects output if ddp is enabled.

        :param output: this output is from training_step
        :return:
        """
        gathered_output = dict(map(lambda i: (i[0], i[1].mean()), outputs.items()))
        self.log_dict(gathered_output)

    @torch.no_grad()
    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        """This method is used for testing.

        :param batch: (features, labels, idx)
            features: (query, positive passage, negative passage)
            labels: answer (score of query-passage, from BEIR)
            idx: index of query-passage pair
        :param batch_idx: index of batch
        :param dataloader_idx: index of dataloader
        :return: {
            "loss": loss,
            "idxs": idxs,
            "relevance": relevance,
            "labels": labels,
            "dataloader_idx": dataloader_idx,
        }
        """
        ids, strings = batch
        is_query = ids[:, 1] == -1
        is_doc = ids[:, 0] == -1

        q_toks = {
            key: strings[key][is_query]
            for key in strings
        }
        d_toks = {
            key: strings[key][is_doc]
            for key in strings
        }
        q_embs = {"encoded_embeddings": []}
        if q_toks["input_ids"].shape[0] > 0:
            q_embs = self.model.encode_query(q_toks)

        d_embs = {"encoded_embeddings": []}
        if d_toks["input_ids"].shape[0] > 0:
            d_embs = self.model.encode_passage(d_toks)
        # emb 여러개 나옴.
        qs = {
            "idxs": ids[is_query][:, 0],
            "embs": q_embs["encoded_embeddings"],
        }
        ds = {
            "idxs": ids[is_doc][:, 1],
            "embs": d_embs["encoded_embeddings"],
        }

        return {
            "save": {
                "qs": qs,
                "ds": ds,
            },
        }

    @rank_zero_only
    def record_map(self):
        ...
        # x_ticks = [str(i) for i in range(self.emb_voc_map_avg.shape[1])]
        # y_ticks = [str(i) for i in range(self.emb_voc_map_avg.shape[0])]
        # emb_voc_map_avg = self.emb_voc_map_avg / self.emb_voc_map_cnt
        # emb_voc_map_square = self.emb_voc_map_square_avg / self.emb_voc_map_cnt
        # emb_voc_map_var = emb_voc_map_square - emb_voc_map_avg ** 2

        # wandb.log({
        #     'avg': wandb.plots.HeatMap(
        #         x_labels=x_ticks, y_labels=y_ticks,
        #         matrix_values=emb_voc_map_avg.cpu().numpy(),
        #     ),
        #     'var': wandb.plots.HeatMap(
        #         x_labels=x_ticks, y_labels=y_ticks,
        #         matrix_values=emb_voc_map_var.cpu().numpy(),
        #     ),
        # })

        # torch.save(emb_voc_map_avg, f"/workspace/tmp/map/avg/{self.cnt}.pt")
        # torch.save(emb_voc_map_var, f"/workspace/tmp/map/var/{self.cnt}.pt")
        #
        # self.emb_voc_map_avg = self.emb_voc_map_avg * 0
        # self.emb_voc_map_square_avg = self.emb_voc_map_square_avg * 0
        # self.emb_voc_map_cnt = 0
        # self.cnt += 1

    def on_validation_epoch_end(self) -> None:
        ...
        # self.record_map()

    def test_step(self, batch: Any, batch_idx: int, **kwargs):
        return self.validation_step(batch, batch_idx, **kwargs)

    def on_test_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int = 0, **kwargs) -> None:
        super().on_test_batch_start(batch, batch_idx, dataloader_idx=0, **kwargs)


class HybridDenseMLMRetrievalModel(HybridRetrievalModel):
    def __init__(self, model: RetrieverBase, sparse: Union[RetrieverBase, nn.Module],
                 train_loss: nn.Module, val_loss: EvaluationMetric, optimizer: Optimizer,
                 scheduler: lr_scheduler):
        super().__init__(model, sparse, train_loss, val_loss, optimizer, scheduler)

    def training_step(self, batch: Any, batch_idx: int) -> Dict[str, Tensor]:
        """

        Args:
            batch:
            batch_idx:
            optimizer_idx:

        Returns:

        """

        _, _, queries, docs, labels = batch
        dense_queries, sparse_queries = queries
        (dense_pos_docs, dense_neg_docs), (sparse_pos_docs, sparse_neg_docs) = docs

        train_queries_emb, train_pos_docs_emb, train_neg_docs_emb = self.dense.encode(
            dense_queries, dense_pos_docs, dense_neg_docs
        )

        train_pos_score = self.dense.score(train_queries_emb, train_pos_docs_emb)
        train_neg_score = self.dense.score(train_queries_emb, train_neg_docs_emb)

        with torch.no_grad():
            teach_queries_emb = self.sparse(train_queries_emb["encoded_matrix"])
            teach_pos_docs_emb = self.sparse(train_pos_docs_emb["encoded_matrix"])
            teach_neg_docs_emb = self.sparse(train_neg_docs_emb["encoded_matrix"])

        self.train_criterion: Union[Dense2SparseMapCriterion, ATDSCriterion]
        losses: Dict[str, Tensor] = self.train_criterion(
            query_emb=train_queries_emb,
            pos_pas_emb=train_pos_docs_emb,
            neg_pas_emb=train_neg_docs_emb,
            teacher_query_emb=teach_queries_emb,
            teacher_pos_pas_emb=teach_pos_docs_emb,
            teacher_neg_pas_emb=teach_neg_docs_emb,
            pos_score_res=train_pos_score,
            neg_score_res=train_neg_score,
            label=labels,
            batch_idx=batch_idx,
        )

        return losses

    def test_step(self, batch: Any, batch_idx: int, **kwargs):
        return self.validation_step(batch, batch_idx, **kwargs)
