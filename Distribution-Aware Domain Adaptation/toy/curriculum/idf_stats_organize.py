import json
from functools import reduce
from pathlib import Path

import pandas as pd
from torch import tensor

from src.datamodule.beir.downloader import BEIRDownloader
from src.datamodule.gpl.downloader import GPLDownloader
from src.datamodule.modules.dataloader import GenericDataLoader
from src.losses.modules.kl_div import KLDivergenceLoss

if __name__ == '__main__':
    dtype = "beir"

    root_dir = Path("/workspace")
    type_dir = root_dir / dtype

    data_dir = type_dir / "msmarco"
    idf_dir = data_dir / "idf"
    msmarco_idfs = tensor(
        json.load(
            (idf_dir / "idfs.json").open("r")
        )
    )

    kl_div = KLDivergenceLoss()

    data_names = [
        "arguana",
        "climate-fever",
        # "bioasq",
        # "signal-1m",
        # "trec-news",
        # "robus04",
        "webis-touche2020",
        # "cqadupstack",
        "dbpedia-entity",
        "fever",
        "fiqa",
        "hotpotqa",
        "nfcorpus",
        "nq",
        "quora",
        "scidocs",
        "scifact",
        "trec-covid",
    ]

    total_rows = []
    for data_name in data_names:
        print(f"Processing {data_name}")

        data_dir = type_dir / data_name
        idf_dir = data_dir / "idf"

        idfs = tensor(
            json.load(
                (idf_dir / "idfs.json").open("r")
            )
        )

        kl_div_idf = kl_div(idfs, msmarco_idfs)
        total_rows.append(
            {
                "idf": data_name,
                "kl_div": kl_div_idf.item(),
            }
        )
    print(total_rows)
