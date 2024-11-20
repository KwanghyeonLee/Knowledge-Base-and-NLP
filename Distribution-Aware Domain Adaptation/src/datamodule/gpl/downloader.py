from src.datamodule.modules.downloader import DataDownloader


class GPLDownloader(DataDownloader):
    def __init__(self, data_dir: str):
        super().__init__(
            "gpl",
            data_dir,
            data_essentials=["gpl-training-data.tsv", "hard-negatives.jsonl", "qgen-qrels"],
        )
