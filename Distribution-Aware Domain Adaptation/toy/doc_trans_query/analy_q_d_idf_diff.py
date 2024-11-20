import json
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

beir_dir = Path("/workspace/beir")

if __name__ == '__main__':
    datasets = [
        "nfcorpus",
        "scifact",
        "trec-covid",
        "scidocs"
    ]

    for dataset in datasets:
        dataset_dir = beir_dir / dataset
        idf_dir = dataset_dir / "idf"

        corpus_idf_file = idf_dir / "d_idf.json"
        query_idf_file = idf_dir / "q_idf.json"

        corpus_idf = json.load(corpus_idf_file.open("r"))
        query_idf = json.load(query_idf_file.open("r"))

        corpus_max = max(corpus_idf)
        query_max = max(query_idf)

        corpus_idf = list(map(lambda i: i if i != corpus_max else 0, corpus_idf))
        query_idf = list(map(lambda i: i if i != query_max else 0, query_idf))

        plt.plot(corpus_idf, label="corpus")
        plt.plot(query_idf, label="query")
        plt.legend()
        plt.title(f"{dataset} idf")
        plt.show()
        plt.close()

        sns.displot(
            {
                "corpus": corpus_idf,
                "query": query_idf
            },
            kind="kde", bw_adjust=.25,
            legend=True,
        ).set(title=f"{dataset} idf distribution")
        plt.show()

        print()

