import os
from pathlib import Path

from src import utils
from src.datamodule.modules.downloader import DataDownloader, SSHDownloader

log = utils.get_pylogger(__name__)


class BEIRDownloader(DataDownloader):
    def __init__(self, data_dir: str):
        super().__init__(
            "beir",
            data_dir,
            data_essentials=["corpus.jsonl", "queries.jsonl", "qrels"],
        )

if __name__ == '__main__':

    for dataset in [
        # "bioasq",
        # "cqadupstack",
        "fiqa",
        # "robust04",
        "scifact",
        "trec-covid-v2"
    ]:
        beir = BEIRDownloader(dataset_name=dataset, data_dir="/workspace/beir")
        beir.download()

