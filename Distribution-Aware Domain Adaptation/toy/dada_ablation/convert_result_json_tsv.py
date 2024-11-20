from pathlib import Path

import pandas as pd
import torch
from tqdm.rich import tqdm

datasets = [
    "nfcorpus",
    "scifact",
    # "trec-covid",
    # "fiqa",
    # "robust04",
    # "scidocs"
]

tmp_dir = Path("/workspace/tmp")

for dataset in datasets:
    result_dir = tmp_dir / dataset / "results"
    result_pth = list(result_dir.glob("*.pt"))[0]

    result = torch.load(result_pth)["results"]
    # ranking.tsv
    ranking_results = list()
    for qid in tqdm(result):
        sorted_result = sorted(
            result[qid].items(), key=lambda x: x[1], reverse=True
        )
        top1000 = sorted_result[:1000]
        for rank, (did, score) in enumerate(top1000):
            ranking_results.append([qid, did, rank + 1, score])

    ranking_df = pd.DataFrame(ranking_results, columns=["qid", "did", "rank", "score"])
    ranking_df.to_csv(result_dir / "ranking.tsv", sep="\t", index=False, header=False)
