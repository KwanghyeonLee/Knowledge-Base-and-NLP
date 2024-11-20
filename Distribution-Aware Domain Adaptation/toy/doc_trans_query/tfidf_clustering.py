from pathlib import Path

from sklearn.decomposition import PCA, SparsePCA
from sklearn.manifold import TSNE

from src.datamodule.modules.dataloader import GenericDataLoader

if __name__ == '__main__':
    # data_name = "trec-covid-v2"
    data_names = [
        # "msmarco",
        # "arguana",
        # "climate-fever",
        # "cqadupstack",
        # "dbpedia-entity",
        # "fever",
        # "robust04",
        # "fiqa",
        # "hotpotqa",
        # "msmarco-v2",
        # "msmarco",
        "nfcorpus",
        # "nq",
        # "quora",
        # "scidocs",
        # "scifact",
        # "trec-covid",
        # "trec-covid-v2",
        # "webis-touche2020",
    ]
    for data_name in data_names:
        print(f"Processing {data_name}")
        model_name = "naver/splade_v2_distil"
        dtype = "beir"

        root_dir = Path("/workspace")
        type_dir = root_dir / dtype
        data_dir = type_dir / data_name

        exp_dir = Path("/workspace/experiments") / data_name
        exp_dir.mkdir(exist_ok=True, parents=True)

        # loader
        train_dataloader = GenericDataLoader(
            data_folder=str(data_dir.absolute()),
            # prefix="qgen",
        )
        # load
        train_corpus, train_corpus_id2_line = train_dataloader.load_custom("corpus")

        # concat title and text
        train_corpus = dict(map(
            lambda k: (k, f"{train_corpus[k]['title']}. {train_corpus[k]['text']}"),
            train_corpus
        ))
        print(f"Loaded {len(train_corpus)} documents")

        # tf-idf
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize

        vectorizer = TfidfVectorizer()
        train_corpus_vec = vectorizer.fit_transform(train_corpus.values())
        train_corpus_vec = normalize(train_corpus_vec)
        print(f"TF-IDF shape: {train_corpus_vec.shape}")

        # cluster
        k = 10
        random_state = 42
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=k, random_state=random_state)
        train_corpus_cluster = kmeans.fit_predict(train_corpus_vec)
        print(f"Cluster shape: {train_corpus_cluster.shape}")

        # save
        import pickle

        with (exp_dir / "train_corpus_vec.pkl").open("wb") as f:
            pickle.dump(train_corpus_vec, f)

        with (exp_dir / "train_corpus_cluster.pkl").open("wb") as f:
            pickle.dump(train_corpus_cluster, f)

        print("Saved.")

        # visualize
        import matplotlib.pyplot as plt

        vis = TSNE(
            n_components=k, random_state=random_state,
            method="exact",
            # perplexity=50,
            # n_iter=100,
            verbose=1
        )
        train_corpus_vec_vis = vis.fit_transform(train_corpus_vec.toarray())

        plt.scatter(
            train_corpus_vec_vis[:, 0], train_corpus_vec_vis[:, 1],
            c=train_corpus_cluster
        )
        plt.show()
        plt.savefig(exp_dir / "train_corpus_vec_tsne.png")
        print("Done.")


