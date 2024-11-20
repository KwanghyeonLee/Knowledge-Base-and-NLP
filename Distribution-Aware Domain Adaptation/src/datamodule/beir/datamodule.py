import json
import linecache
from collections import defaultdict
from dataclasses import dataclass
from itertools import chain, product
from os import PathLike
from pathlib import Path
from typing import Dict, List, Optional, Union

import orjson
import torch
from torch.utils.data import Dataset

from src import utils
from src.datamodule.utils import EvalMode, concat_title_and_body, InputData

log = utils.get_pylogger(__name__)


class BEIRDataset(Dataset):
    def __init__(
            self,
            queries: dict,
            corpus: Union[dict, str, PathLike],
            qrels: dict,
            rerank: Optional[Dict[str, List[str]]],
            mode: EvalMode,
            sep=" ",
            corpusid2line: Optional[Dict[str, int]] = None,
            corpus_path: Optional[Union[str, PathLike, Path]] = None,
    ):
        self.queries = queries

        if type(corpus) is not dict and (corpusid2line is None or corpus_path is None):
            raise ValueError("corpusid2line must be provided when corpus is not a dict.")
        self.corpus = corpus
        self.corpusid2line = corpusid2line
        self.corpus_path = corpus_path

        self.rerank = rerank
        self.mode = mode

        remove_qids = []
        remove_pids = defaultdict(list)
        for qid in qrels:
            for pid in qrels[qid]:
                if qid not in self.queries:
                    remove_qids.append(qid)
                if pid not in self.corpus:
                    remove_pids[qid].append(pid)

        for qid in remove_qids:
            del qrels[qid]

        for qid in remove_pids:
            for pid in remove_pids[qid]:
                del qrels[qid][pid]

        self.queries2id = {v: k for k, v in enumerate(self.queries.keys())}
        self.qidx2qid = {v: k for k, v in self.queries2id.items()}
        if len(self.corpus) > 0:
            self.corpus2id = {v: k for k, v in enumerate(self.corpus.keys())}
        else:
            self.corpus2id = self.corpusid2line
        self.pidx2pid = {v: k for k, v in self.corpus2id.items()}

        # if only re-rank is selected
        if mode.re_rank is True and mode.full_rank is False:
            log.info("Evaluation by reranking tas-b colbert's first stage ranking.")
            self.possible_combs = list(
                chain.from_iterable(
                    map(
                        lambda qid: map(lambda pid: (qid, pid), self.rerank[qid]),
                        self.rerank.keys(),
                    )
                )
            )
        else:
            # self.possible_combs = list(product(queries.keys(), corpus.keys()))
            self.possible_combs = [
                *list(map(
                    lambda key: ("q", key),
                    self.queries2id.keys()
                )),
                *list(map(
                    lambda key: ("d", key),
                    self.corpus2id.keys()
                ))
            ]

        log.info(f"Evaluation QD pairs: {len(self.possible_combs)}")

        self.sep = sep

        self.qrels = torch.zeros((len(self.queries2id), len(self.corpus2id)), dtype=torch.int8)
        for qid in qrels:
            for pid in qrels[qid]:
                self.qrels[self.queries2id[qid], self.corpus2id[pid]] = qrels[qid][pid]

    def is_rerank_target(self, qid, did):
        if self.mode.re_rank is False:
            return False

        if self.rerank is None:
            return False

        return (qid in self.rerank) and (did in self.rerank[qid])

    def __getitem__(self, item):
        index = item
        dtype, val = self.possible_combs[index]

        if dtype == "q":
            qid = val
            query_text = self.queries[qid]
            return InputData(
                rerank=False if self.mode.re_rank is False else True,
                guid=(self.queries2id[qid], -1),
                query=query_text,
                doc="",
                label=0,
            )

        elif dtype == "d":
            pid = val
            if len(self.corpus) > 0:
                passage_text = concat_title_and_body(pid, self.corpus, self.sep)
            else:
                file_name = self.corpus_path
                lineno = self.corpusid2line[pid] + 1
                line = linecache.getline(file_name, lineno, module_globals=None)
                json_line = orjson.loads(line)

                corpus_dict = {
                    pid: json_line
                }
                passage_text = concat_title_and_body(pid, corpus_dict, self.sep)

            return InputData(
                rerank=False if self.mode.re_rank is False else True,
                guid=(-1, self.corpus2id[pid]),
                query="",
                doc=passage_text,
                label=0,
            )

    def __len__(self):
        # return 2_000
        return len(self.possible_combs)


class BEIRTrainDataset(Dataset):
    def __init__(
        self,
        queries: dict,
        corpus: dict,
        qrels: dict,
        rerank: Optional[Dict[str, List[str]]],
        mode: EvalMode,
        sep=" ",
    ):
        self.queries = queries
        self.corpus = corpus
        self.qrels = qrels
        self.rerank = rerank
        self.mode = mode

        self.possible_combs = [
            (qid, pid)
            for qid in self.qrels
            for pid in self.qrels[qid]
        ]
        # self.possible_combs = list(product(queries.keys(), corpus.keys()))
        log.info(f"Evaluation QD pairs: {len(self.possible_combs)}")

        self.sep = sep

        self.queries2id = {v: k for k, v in enumerate(self.queries.keys())}
        self.corpus2id = {v: k for k, v in enumerate(self.corpus.keys())}

    def is_rerank_target(self, qid, did):
        if self.mode.re_rank is False:
            return False

        if self.rerank is None:
            return False

        return (qid in self.rerank) and (did in self.rerank[qid])

    def __getitem__(self, item):
        index = item
        qid, pid = self.possible_combs[index]
        query_text = self.queries[qid]
        passage_text = concat_title_and_body(pid, self.corpus, self.sep)

        label = 0
        if qid in self.qrels:
            if pid in self.qrels[str(qid)]:
                label = self.qrels[str(qid)][str(pid)]

        return InputData(
            rerank=self.is_rerank_target(qid, pid),
            guid=(self.queries2id[qid], self.corpus2id[pid]),
            query=query_text,
            doc=passage_text,
            label=label,
        )

    def __len__(self):
        return len(self.possible_combs)

