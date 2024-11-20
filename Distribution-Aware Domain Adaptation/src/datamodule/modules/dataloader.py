import csv
import logging
import os
from typing import Dict, List, Tuple, Union

import orjson
from tqdm.rich import tqdm

logger = logging.getLogger(__name__)



class GenericDataLoader:
    def __init__(
        self,
        data_folder: str = None,
        prefix: str = None,
        corpus_file: str = "corpus.jsonl",
        query_file: str = "queries.jsonl",
        qrels_folder: str = "qrels",
        qrels_file: str = None,
        rank: int = None,
    ):
        self.rank = rank

        self.corpus = {}
        self.corpusid2line = {}
        self.queries = {}
        self.qrels = {}

        if prefix:
            query_file = prefix + "-" + query_file
            qrels_folder = prefix + "-" + qrels_folder

        self.corpus_file = os.path.join(data_folder, corpus_file) if data_folder else corpus_file
        self.query_file = os.path.join(data_folder, query_file) if data_folder else query_file
        self.qrels_folder = os.path.join(data_folder, qrels_folder) if data_folder else None
        self.qrels_file = os.path.join(data_folder, qrels_file) if qrels_file and data_folder else None

    @staticmethod
    def check(fIn: str, ext: str):
        if not os.path.exists(fIn):
            raise ValueError(f"File {fIn} not present! Please provide accurate file.")

        if not fIn.endswith(ext):
            raise ValueError(f"File {fIn} must be present with extension {ext}")

    def load_custom(
        self,
        which: str,
        split: str = None,
    ) -> Union[Dict[str, str], Tuple[Dict[str, str], Dict[str, int]]]:
        if which == "corpus":
            self.check(fIn=self.corpus_file, ext="jsonl")
            if not len(self.corpus):
                self._load_corpus()
            return self.corpus, self.corpusid2line

        elif which == "queries":
            self.check(fIn=self.query_file, ext="jsonl")
            if not len(self.queries):
                self._load_queries()
            return self.queries

        elif which == "qrels":
            if split is None:
                raise ValueError("specify split")
            if self.qrels_file is None:
                self.qrels_file = os.path.join(self.qrels_folder, split + ".tsv")

            self.check(fIn=self.qrels_file, ext="tsv")
            if os.path.exists(self.qrels_file):
                self._load_qrels()
            return self.qrels
        else:
            raise ValueError("which must be either `corpus`, `queries`, `qrels`")

    def load(
        self, split="test"
    ) -> Tuple[
        Tuple[Dict[str, Dict[str, str]], Dict[str, int]],
        Dict[str, str],
        Dict[str, Dict[str, int]]]:
        self.qrels_file = os.path.join(self.qrels_folder, split + ".tsv")
        self.check(fIn=self.corpus_file, ext="jsonl")
        self.check(fIn=self.query_file, ext="jsonl")
        self.check(fIn=self.qrels_file, ext="tsv")

        if not len(self.corpus):
            self._load_corpus()

        if not len(self.queries):
            self._load_queries()

        if os.path.exists(self.qrels_file):
            self._load_qrels()
            self.queries = {qid: self.queries[qid] for qid in self.qrels}

        return (self.corpus, self.corpusid2line), self.queries, self.qrels

    def load_corpus(self) -> Dict[str, Dict[str, str]]:
        self.check(fIn=self.corpus_file, ext="jsonl")

        if not len(self.corpus):
            self._load_corpus()

        return self.corpus

    def _load_corpus(self):
        num_lines = len(open(self.corpus_file, "rb").readlines())
        with open(self.corpus_file, encoding="utf8") as fIn:
            iter_target = tqdm(
                fIn,
                total=num_lines,
                desc="Loading Corpus",
                unit_scale=1000000,
            ) if self.rank == 0 else fIn
            for line_number, line in enumerate(iter_target):
                line = orjson.loads(line)
                if num_lines >= 10_000_000:
                    # only the case for bioasq
                    self.corpusid2line[line.get("_id")] = line_number
                else:
                    self.corpus[line.get("_id")] = {
                        "text": line.get("text"),
                        "title": line.get("title"),
                    }

    def _load_queries(self):
        num_lines = len(open(self.query_file, "rb").readlines())
        with open(self.query_file, encoding="utf8") as fIn:
            iter_target = tqdm(
                fIn,
                total=num_lines,
                desc="Loading Queries",
                unit_scale=1000000000,
            ) if self.rank == 0 else fIn
            for line in iter_target:
                line = orjson.loads(line)
                self.queries[line.get("_id")] = line.get("text")

    def _load_qrels(self):
        reader = csv.reader(
            open(self.qrels_file, encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_MINIMAL
        )
        next(reader)

        num_lines = len(open(self.qrels_file, "rb").readlines())
        iter_target = tqdm(
            reader,
            total=num_lines,
            desc="Loading Qrels",
            unit_scale=1000000000,
        ) if self.rank == 0 else reader
        for row in iter_target:
            query_id, corpus_id, score = row[0], row[1], int(row[2])

            if query_id not in self.qrels:
                self.qrels[query_id] = {corpus_id: score}
            else:
                self.qrels[query_id][corpus_id] = score
