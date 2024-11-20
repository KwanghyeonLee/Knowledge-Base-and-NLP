from pathlib import Path

import pandas as pd

workspace = Path("/workspace")
ramda_dir = workspace / "ramda"
tmp_dir = workspace / "tmp"

datasets = [
    # "scifact",
    "trec-covid",
    # "fiqa",
    # "robust04",
    # "nfcorpus", "scidocs"
]

ranking_tsv_file_name = "ranking.tsv"

for dataset in datasets:
    if dataset == "trec-covid":
        ramda_rank = ramda_dir / "trec-covid-v2"
    else:
        ramda_rank = ramda_dir / dataset
    ramda_rank /= ranking_tsv_file_name

    ours_rank = tmp_dir / dataset / "results" / ranking_tsv_file_name

    ramda_df = pd.read_csv(ramda_rank, sep="\t", names=["qid", "pid", "rank", "score"])
    ours_df = pd.read_csv(ours_rank, sep="\t", names=["qid", "pid", "rank", "score"])

    print()




