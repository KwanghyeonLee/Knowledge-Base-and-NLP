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
