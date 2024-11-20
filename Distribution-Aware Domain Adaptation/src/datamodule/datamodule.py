import json
import os
from collections import Counter
from functools import reduce
from pathlib import Path

from typing import Dict, List, Optional, Union, Tuple

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch import Tensor, tensor
from torch.utils.data import DataLoader

from src.data2tensor.hybrid import HybridTokenizer
from src.data2tensor.tokenizer import BaseTokenizer
from src.datamodule.beir.datamodule import BEIRDataset, BEIRTrainDataset
from src.datamodule.beir.downloader import BEIRDownloader
from src.datamodule.gpl.datamodule import GenerativePseudoLabelingDataset
from src.datamodule.gpl.downloader import GPLDownloader
from src.datamodule.modules.dataloader import GenericDataLoader
from src.datamodule.utils import InputData, EvalMode, load_ranking


class RetrievalDataModule(pl.LightningDataModule):
    def __init__(
            self,
            train_dataset_type: str = "beir",
            train_dataset: str = "msmarco",
            test_dataset_type: str = "beir",
            test_datasets: List[str] = None,
            shuffle_train: bool = False,
            dada_dir: str = None,
            hyde_dir: str = None,
            m2m_data_dir: str = None,
            gpl_data_dir: str = None,
            beir_data_dir: str = None,
            ramda_data_dir: str = None,
            rerank_data_dir: str = None,
            rerank_topk: int = 100,
            rerank: bool = False,
            tokenizer: Union[BaseTokenizer, HybridTokenizer] = None,
            train_max_step: int = 140_000,
            train_batch_size: int = 32,
            test_batch_size: int = 64,
            workers: int = 64,
            curriculum: str = None,
    ):
        super().__init__()
        self.curriculum = curriculum
        if self.curriculum is None:
            self.curriculum = "gpl-training-data.tsv"

        self.train_dataset_type = train_dataset_type
        self.train_dataset = train_dataset
        self.test_dataset_type = test_dataset_type
        self.test_datasets = test_datasets

        if self.train_dataset_type not in ["beir", "gpl", "hyde", "m2m", "ramda", "dada"]:
            raise ValueError("Specify the train data type! (beir, gpl, hyde)")
        if self.test_dataset_type not in ["beir", "gpl", "hyde", "m2m", "ramda", "dada"]:
            raise ValueError("Specify the test data type! (beir, gpl, hyde)")

        if self.train_dataset is None:
            raise ValueError("Specify the train data name!")
        if self.test_datasets is None:
            raise ValueError("Specify the test data name!")

        self.shuffle_train = shuffle_train
        self.train_max_step = train_max_step
        self.test_data = None
        self.train_data = None
        self.val_data = None
        self.tokenizer = tokenizer
        self.gpl_data_dir = gpl_data_dir
        self.beir_data_dir = beir_data_dir
        self.hyde_data_dir = hyde_dir
        self.m2m_data_dir = m2m_data_dir
        self.ramda_data_dir = ramda_data_dir
        self.dada_dir = dada_dir

        self.rerank = rerank
        self.eval_mode = self._get_mode()
        if self.eval_mode.re_rank:
            self.rerank_data_dir = Path(rerank_data_dir)
            self.rerank_topk = rerank_topk

        self.qgen_prefix = "qgen"

        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size

        self.workers = workers

        self.beir_downloader = BEIRDownloader(data_dir=self.beir_data_dir)
        self.gpl_downloader = GPLDownloader(data_dir=self.gpl_data_dir)

    def _get_mode(self):
        """This method is used for checking either the evaluation rerank or full-rank.

        :return:
            True: rerank enabled, False: full-rank mode.
        """
        return EvalMode(re_rank=True) if self.rerank else EvalMode(re_rank=False, full_rank=True)

    def _get_downloader(self, dtype: str):
        if dtype == "beir":
            return self.beir_downloader
        elif dtype == "gpl":
            return self.gpl_downloader

    def prepare_data(self) -> None:
        """This method prepares selected dataset from GPL and BEIR dataset url on desired path."""
        download_targets = [
            # TRAIN-DATASET
            (self.train_dataset_type, self.train_dataset),
            # EVAL-DATASET
            *[
                (self.test_dataset_type, eval_dataset)
                for eval_dataset in self.test_datasets
            ]
        ]
        for dtype, dataset_name in download_targets:
            if dtype in ["m2m", "ramda", "dada", "hyde"]:  # m2m is manually downloaded
                continue

            downloader = self._get_downloader(dtype)
            downloader.download(dataset_name=dataset_name)

    def _prepare_data(
            self,
            dtype: str = "beir",
            split: str = None,
            dataset_name: str = None,
    ):
        if split is None:
            raise ValueError("set split: `train`, `test`, `dev`")

        if dataset_name is None:
            raise ValueError("specify the dataset name")

        # loading corpus, queries and qrels
        corpus, queries, qrels = None, None, None

        data_folder = {
            "beir": self.beir_data_dir,
            "gpl": self.gpl_data_dir,
            "m2m": self.m2m_data_dir,
            "ramda": self.ramda_data_dir,
            "dada": self.dada_dir,
            "hyde": self.hyde_data_dir
        }[dtype]

        data_folder += f"/{dataset_name}"
        prefix = self.qgen_prefix if dtype in ["gpl", "ramda", "dada", "hyde"] else None

        (corpus, corpusid2line), queries, qrels = GenericDataLoader(
            data_folder=data_folder,
            prefix=prefix,
            rank=self.trainer.local_rank
        ).load(split=split)

        rerank = None
        if (split == "test" or split == "validate") and self.eval_mode.re_rank:
            rerank = load_ranking(
                path=self.rerank_data_dir / dataset_name / f"{dataset_name}.ranking.tsv",
                topk=self.rerank_topk,
            )

        if dtype == "beir":
            return BEIRDataset(
                queries=queries,
                corpus=corpus,
                corpusid2line=corpusid2line,
                corpus_path=os.path.join(data_folder, "corpus.jsonl"),
                qrels=qrels,
                rerank=rerank,
                mode=self.eval_mode,

            )
        elif dtype in ["gpl", "ramda", "dada", "hyde"]:
            tsv_file = self.curriculum
            return GenerativePseudoLabelingDataset(
                data_dir=data_folder,
                queries=queries,
                corpus=corpus,
                qrels=qrels,
                tsv_file=tsv_file,
            )

        elif dtype == "m2m":
            return BEIRTrainDataset(
                queries=queries,
                corpus=corpus,
                # corpusid2line=corpusid2line,
                # corpus_path=os.path.join(data_folder, "corpus.jsonl"),
                qrels=qrels,
                rerank=rerank,
                mode=self.eval_mode,

            )

    def setup(self, stage: Optional[str] = None) -> None:
        """this method setups dataset on every gpu if ddp is enabled.

        :param stage: ["fit", "validate", "test", "predict"]
        """
        if stage not in ["fit", "validate", "test", "predict"]:
            raise ValueError("stage should be either `fit`, `test`, `predict`.")

        if stage == "fit":
            self.train_data = self._prepare_data(
                dtype=self.train_dataset_type,
                dataset_name=self.train_dataset,
                split="train",
            )
            self.val_data = {
                test_dataset: self._prepare_data(
                    dtype=self.test_dataset_type,
                    split="test",
                    dataset_name=test_dataset,
                )
                for test_dataset in self.test_datasets
            }

        if stage == "validate":
            self.val_data = {
                test_dataset: self._prepare_data(
                    dtype=self.test_dataset_type,
                    split="test",
                    dataset_name=test_dataset,
                )
                for test_dataset in self.test_datasets
            }

        if stage == "test":
            if self.val_data is not None:
                self.test_data = self.val_data
            else:
                self.test_data = {
                    test_dataset: self._prepare_data(
                        dtype=self.test_dataset_type,
                        split="test",
                        dataset_name=test_dataset,
                    )
                    for test_dataset in self.test_datasets
                }

    def smart_batching_collate(self, batch: List[InputData]):
        """Transforms a batch from a SmartBatchingDataset to a batch of tensors for the model Here,
        batch is a list of tuples: [(tokens, label), ...]

        :param batch:
            a batch from a SmartBatchingDataset
        :return:
            a batch of tensors for the model
        """
        is_eval = self.trainer.training is False
        if is_eval:
            if type(self.tokenizer) == HybridTokenizer:
                # this is for ATDS
                strings: List[str] = []
                ids: Union[List[Tuple[int, int]], Tensor] = []

                for example in batch:
                    strings.append(example.query if example.doc == "" else example.doc)
                    ids.append(example.guid)

                ids = torch.tensor(ids)

                strings_tokenized: Dict[str, Union[Tensor, any]] = self.tokenizer.dense.tokenize_passage(
                    strings
                )
                return ids, strings_tokenized

            else:
                # this is ordinary
                strings: List[str] = []
                ids: Union[List[Tuple[int, int]], Tensor] = []

                for example in batch:
                    strings.append(example.query if example.doc == "" else example.doc)
                    ids.append(example.guid)

                ids = torch.tensor(ids)

                strings_tokenized: Dict[str, Union[Tensor, any]] = self.tokenizer.tokenize_passage(
                    strings
                )
                return ids, strings_tokenized

        else:
            # this is for training...
            queries: List[str] = []
            docs: List[str, List[str]] = []
            labels: Union[List[int], Tensor] = []
            ids: Union[List[Tuple[int, int]], Tensor] = []
            is_rerank: Union[List[int], Tensor] = []

            for example in batch:
                queries.append(example.query)
                docs.append(example.doc)
                labels.append(example.label)
                ids.append(example.guid)
                is_rerank.append(1 if example.rerank else 0)

            labels = torch.tensor(labels)
            # ids = torch.tensor(ids)
            is_rerank = torch.tensor(is_rerank)

            if type(self.tokenizer) == HybridTokenizer:
                # this is for ATDS
                dense_query_tokenized: Dict[str, Union[Tensor, any]] = self.tokenizer.dense.tokenize_query(queries)
                sparse_query_tokenized: Optional[Dict[str, Union[Tensor, any]]] = None
                sparse_docs_tokenized: Optional[Dict[str, Union[Tensor, any]]] = None

                sparse_query_tokenized = self.tokenizer.sparse.tokenize_query(queries)
                dense_docs_tokenized: Dict[str, Union[Tensor, any]] = self.tokenizer.dense.tokenize_passage(
                    list(reduce(lambda i, j: i + j, zip(*docs)))
                )
                sparse_docs_tokenized: Dict[str, Union[Tensor, any]] = self.tokenizer.sparse.tokenize_passage(
                    list(reduce(lambda i, j: i + j, zip(*docs)))
                )
                dense_docs_tokenized: List[Dict[str, Union[Tensor, any]]] = list(
                    map(
                        lambda d: dict(zip(dense_docs_tokenized.keys(), d)),
                        zip(
                            *map(
                                lambda k: dense_docs_tokenized[k].tensor_split(2, dim=0),
                                dense_docs_tokenized.keys(),
                            )
                        ),
                    )
                )
                sparse_docs_tokenized: List[Dict[str, Union[Tensor, any]]] = list(
                    map(
                        lambda d: dict(zip(sparse_docs_tokenized.keys(), d)),
                        zip(
                            *map(
                                lambda k: sparse_docs_tokenized[k].tensor_split(2, dim=0),
                                sparse_docs_tokenized.keys(),
                            )
                        ),
                    )
                )
                return (is_rerank, ids,
                        (dense_query_tokenized, sparse_query_tokenized),
                        (dense_docs_tokenized, sparse_docs_tokenized),
                        labels)
            else:
                # this is ordinary gpl training
                query_tokenized: Dict[str, Union[Tensor, any]] = self.tokenizer.tokenize_query(queries)
                if type(docs[0]) != list:
                    docs_tokenized: Dict[str, Union[Tensor, any]] = self.tokenizer.tokenize_passage(docs)

                else:
                    docs_tokenized: Dict[str, Union[Tensor, any]] = self.tokenizer.tokenize_passage(
                        list(reduce(lambda i, j: i + j, zip(*docs)))
                    )
                    docs_tokenized: List[Dict[str, Union[Tensor, any]]] = list(
                        map(
                            lambda d: dict(zip(docs_tokenized.keys(), d)),
                            zip(
                                *map(
                                    lambda k: docs_tokenized[k].tensor_split(2, dim=0),
                                    docs_tokenized.keys(),
                                )
                            ),
                        )
                    )
                return is_rerank, ids, query_tokenized, docs_tokenized, labels

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        # Here shuffle=False, since (or assuming) we have done it in the pseudo labeling
        return DataLoader(
            self.train_data,
            collate_fn=self.smart_batching_collate,
            batch_size=self.train_batch_size,
            shuffle=self.shuffle_train,
            drop_last=False,
            num_workers=self.workers,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return [
            DataLoader(
                self.val_data[val_dataset],
                collate_fn=self.smart_batching_collate,
                batch_size=self.test_batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=self.workers,
            )
            for val_dataset in self.val_data
        ]

    def test_dataloader(self) -> EVAL_DATALOADERS:
        # dont shuffle because this is test dataset
        return [
            DataLoader(
                self.test_data[test_dataset],
                collate_fn=self.smart_batching_collate,
                batch_size=self.test_batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=self.workers,
            )
            for test_dataset in self.test_datasets
        ]

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return self.val_dataloader()
