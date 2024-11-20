from pathlib import Path

import pandas as pd
import torch
from scipy.stats import ttest_rel, ttest_ind

perquery_dir = Path("/workspace") / "research" / "perquery"

for model in [
    # "cond",
    "coco",
    "tasb",
]:
    total = []
    for dataset in [
        "scifact",
        "nfcorpus",
        "fiqa",
        "scidocs",
        "robust04"
    ]:
        if not (perquery_dir / model / "gpl" / f"{dataset}.pt").exists():
            continue

        gpl = (
            pd.DataFrame(
                torch.load(perquery_dir / model / "gpl" / f"{dataset}.pt")
            ).transpose()["fl.ndcg"].to_frame()
        )
        gpl["ndcg"] = gpl["fl.ndcg"].apply(lambda x: x["NDCG@10"])
        gpl.drop(columns=["fl.ndcg"], inplace=True)
        gpl = gpl.reset_index().rename(columns={"index": "qid"})
        dada = (
            pd.DataFrame(
                torch.load(perquery_dir / model / "dada" / f"{dataset}.pt")
            ).transpose()["fl.ndcg"].to_frame()
        )
        dada["ndcg"] = dada["fl.ndcg"].apply(lambda x: x["NDCG@10"])
        dada.drop(columns=["fl.ndcg"], inplace=True)
        dada = dada.reset_index().rename(columns={"index": "qid"})
        df = pd.merge(
            gpl, dada, on="qid", suffixes=("_gpl", "_dada")
        )

    #     total.append(df)
    #
    # total_df = pd.concat(total)

        # paired t-test
        t, p = ttest_rel(df["ndcg_gpl"], df["ndcg_dada"])
        print(f"{model} {dataset} p-value: {p:.4f}")

        # student t-test
        t, p = ttest_ind(df["ndcg_gpl"], df["ndcg_dada"])
        print(f"{model} {dataset} p-value: {p:.4f}")



        # mean value
        # print(f"{model} {dataset} mean NDCG@10 GPL: {df['ndcg_gpl'].mean():.4f}, DADA: {df['ndcg_dada'].mean():.4f}")

