import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

from nltk import TreebankWordDetokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from torch import Tensor, device
from tqdm import tqdm

from src import utils

STOPWORDS = set(stopwords.words("english"))
stemmer = PorterStemmer()
nltk_detokenizer = TreebankWordDetokenizer()

log = utils.get_pylogger(__name__)


@dataclass
class EvalMode:
    full_rank: bool = None
    re_rank: bool = None

    def __post_init__(self):
        # init
        if self.full_rank is None:
            self.full_rank = False
        if self.re_rank is None:
            self.re_rank = False

        # value check
        if self.full_rank is False and self.re_rank is False:
            log.info("None of EvalMode selected...")
            log.info("EvalMode set as full-rank.")
            self.full_rank = True


def batch_to_device(batch, target_device: device):
    """send a pytorch batch to a device (CPU/GPU)"""
    for key in batch:
        if isinstance(batch[key], Tensor):
            batch[key] = batch[key].to(target_device)
    return batch


def concat_title_and_body(did, corpus, sep):
    document = []
    if did not in corpus:
        print(f"Document {did} not found in corpus.")
    title = corpus[did]["title"].strip()
    body = corpus[did]["text"].strip()
    if len(title):
        document.append(title)
    if len(body):
        document.append(body)
    return sep.join(document)


def load_ranking(path, topk=100):
    ranking = {}
    with open(path, encoding="utf-8") as file:
        for line in file:
            qid, pid, rank, score = line.strip().split("\t")

            rank = int(rank)
            score = float(score)

            if rank <= topk:
                ranking[qid] = ranking.get(qid, set())
                ranking[qid].add(pid)

    for qid in ranking:
        ranking[qid] = list(ranking[qid])

    return ranking


def get_query_term_counts(queries: Dict[str, str]) -> Counter:
    queries_text = queries.values()
    query_term_counts = Counter()
    for query_text in tqdm(
        queries_text,
        total=len(queries_text),
        desc="Calc q_term counts",
        unit_scale=1_000_000_000,
    ):
        query_terms = get_terms(sentence=query_text)
        query_term_counts.update(set(query_terms))

    return query_term_counts


@dataclass
class InputData:
    """Structure for one input example with texts, the label and a unique id."""
    guid: Tuple[int, int] = (-1, -1), # (qid, did)
    query: str = None
    doc: Union[str, List[str]] = None
    label: Union[int, float] = 0

    rerank: bool = False


def get_terms(sentence: str, remove_stopwords: bool = True, stemming: bool = False):
    sentence = re.sub(pattern=r"[^\w\s]", repl="", string=sentence).strip()
    terms = sentence.strip().lower().split()
    if remove_stopwords:
        terms = list(filter(lambda x: x not in STOPWORDS, terms))
    if stemming:
        terms = list(map(stemmer.stem, terms))
    return [_ for _ in terms if _]


def remove_special_tokens(sentence: str, special_tokens: List[str]):
    remove_target = "|".join(special_tokens)
    removed_text = re.sub(pattern=rf"{remove_target}", repl="", string=sentence).strip()
    return removed_text


def detokenize(terms: List[str]):
    return nltk_detokenizer.detokenize(terms)
