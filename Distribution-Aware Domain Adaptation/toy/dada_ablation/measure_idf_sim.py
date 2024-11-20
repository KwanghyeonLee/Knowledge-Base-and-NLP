from collections import Counter
from itertools import combinations
from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from torch import tensor
import matplotlib.ticker as tkr

from src.losses.modules.kl_div import KLDivergenceLoss
from toy.dada_ablation.amnesia.measure_amnesia import load_idf, measure_jaccard_distance

workspace_dir = Path("/workspace")
amnesia_dir = workspace_dir / "amnesia"

kl_div = KLDivergenceLoss()

def extract_topk(idf_score, k=1000):
    a = Counter(idf_score)
    idf_score = sorted(enumerate(idf_score), key=lambda x: x[1], reverse=True)[:k]
    idf_token_ids = set([x[0] for x in idf_score])
    return idf_token_ids


def measure_similarity(idf_a, idf_b):
    # sim = kl_div(
    #     tensor(idf_a),
    #     tensor(idf_b)
    # )
    M = 0.5 * (tensor(idf_a) + tensor(idf_b))
    P = tensor(idf_a)
    Q = tensor(idf_b)
    sim = 0.5 * (kl_div(P, M) + kl_div(Q, M))
    return sim * 10e7


if __name__ == '__main__':
    datasets = [
        "msmarco",
        "nfcorpus",
        "scifact",
        "scidocs",
        "fiqa",
        "robust04"
    ]

    idfs = {}
    for dataset in datasets:
        idf = load_idf("ours-gpl", dataset)
        idfs[dataset] = idf

    idf_sim_df = pd.DataFrame(columns=datasets, index=datasets)
    dataset_combs = combinations(idfs.items(), 2)
    for dataset_comb in dataset_combs:
        dataset_a, dataset_b = dataset_comb
        name_a, name_b = dataset_a[0], dataset_b[0]
        idf_a, idf_b = dataset_a[1], dataset_b[1]

        # idf_a, idf_b = extract_topk(idf_a, k = 100), extract_topk(idf_b, k = 100)
        # jaccard_sim = measure_jaccard_distance(idf_a, idf_b)
        sim = measure_similarity(idf_a, idf_b)
        idf_sim_df.loc[name_b, name_a] = sim.item()


    idf_sim_df.dropna(axis=0, how="all", inplace=True)
    idf_sim_df.dropna(axis=1, how="all", inplace=True)
    idf_sim_df.fillna(0, inplace=True)
    print(idf_sim_df)

    mask = idf_sim_df == 0

    # formatter = tkr.ScalarFormatter(useMathText=True)
    # formatter.set_scientific(True)
    # formatter.set_powerlimits((-2, 2))

    ax = sns.heatmap(
        idf_sim_df, mask=mask, vmin=0, annot=True, fmt=".3g", cmap="crest",
        cbar_kws={'format': '%.2g'}
    )
    cbar = ax.collections[0].colorbar
    cbar.ax.set_title('1e-7')

    plt.title("IDF Similarity")

    plt.show()

