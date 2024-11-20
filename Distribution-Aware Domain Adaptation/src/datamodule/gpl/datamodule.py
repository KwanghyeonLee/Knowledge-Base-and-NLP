import linecache

from torch.utils.data import Dataset

from src.datamodule.modules.dataloader import logger
from src.datamodule.utils import InputData, concat_title_and_body


class GenerativePseudoLabelingDataset(Dataset):

    # def __init__(self, tsv_path, queries, corpus, sep=' '):
    def __init__(self,
                 data_dir,
                 queries, corpus, qrels,
                 sep=" ",
                 tsv_file=None):
        if tsv_file is None:
            tsv_file = "gpl-training-data.tsv"

        self.tsv_path = f"{data_dir}/{tsv_file}"
        self.queries = queries
        self.corpus = corpus
        self.sep = sep
        self.qrels = qrels
        self.ntuples = len(linecache.getlines(self.tsv_path))

        logger.info(f"Loading {self.tsv_path} ...")

    def __getitem__(self, index):
        # index = item + 1  # first row is column name.
        tsv_line = linecache.getline(self.tsv_path, index + 1)
        qid, pos_pid, neg_pid, label = tsv_line.strip().split("\t")
        query_text = self.queries[qid]
        pos_text = concat_title_and_body(pos_pid, self.corpus, self.sep)
        neg_text = concat_title_and_body(neg_pid, self.corpus, self.sep)
        label = float(label)  # CE margin between (query, pos) and (query, neg)

        return InputData(
            guid=(pos_pid, neg_pid),
            query=query_text, doc=[pos_text, neg_text], label=label
        )

    def __len__(self):
        return self.ntuples
