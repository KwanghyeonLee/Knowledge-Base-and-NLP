from itertools import chain

from torch.utils.data import Dataset

from src import utils
from src.datamodule.utils import InputData, concat_title_and_body, detokenize, get_terms

log = utils.get_pylogger(__name__)


class BEIRDataset(Dataset):
    def __init__(
        self,
        queries: dict,
        corpus: dict,
        qrels: dict,
        sep=" ",
        training: bool = True,
    ):
        self.training = training

        self.queries = queries
        self.corpus = corpus
        self.qrels = qrels

        # self.possible_combs = list(self.corpus.keys())

        self.possible_combs = list(zip(self.queries.keys(), self.corpus.keys()))

        log.info(f"Evaluation QD pairs: {len(self.possible_combs)}")

        self.sep = sep

        self.queries2id = {v: k for k, v in enumerate(self.queries.keys())}
        self.corpus2id = {v: k for k, v in enumerate(self.corpus.keys())}

    def get_qrel(self, qid: str, did: str):
        if qid not in self.qrels:
            return 0
        if did not in self.qrels[qid]:
            return 0
        if self.qrels[qid][did] <= 0:
            return 0

        return 1

    def __getitem__(self, index):
        qid, did = self.possible_combs[index]

        query_text: str = self.queries[qid]
        doc_text: str = concat_title_and_body(did, self.corpus, self.sep)
        label = self.get_qrel(qid, did)

        return InputData(
            qid=qid,
            did=did,
            query=query_text,
            doc=doc_text,
            label=label,
        )

    def __len__(self):
        return len(self.possible_combs)
