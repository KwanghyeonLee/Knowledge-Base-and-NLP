import json
from pathlib import Path

import transformers
from numpy import inf
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from tqdm.rich import tqdm

from src.datamodule.modules.dataloader import GenericDataLoader

beir_dir = Path("/workspace/beir")


if __name__ == '__main__':
    datasets = [
        "nfcorpus",
        "scifact",
        "trec-covid",
        "scidocs"
    ]

    tokenizer = transformers.AutoTokenizer.from_pretrained("distilbert-base-uncased")
    vocab_dict = sorted(tokenizer.vocab.items(), key=lambda x: x[1])
    vocab_dict_rev = dict(map(lambda i: (i[1], i[0]), vocab_dict))
    vocab_tokens = list(map(lambda i: i[0], vocab_dict))
    vocab_ids = list(map(lambda i: str(i[1]), vocab_dict))

    for dataset in datasets:
        dataset_dir = beir_dir / dataset

        # loader
        train_dataloader = GenericDataLoader(
            data_folder=str(dataset_dir.absolute()),
        )

        # ============ Corpus ============ #
        pipe = Pipeline([
            ('count', CountVectorizer(vocabulary=vocab_ids)),
            ('tfidf', TfidfTransformer(smooth_idf=False))
        ])
        # load
        train_corpus, train_corpus_id2_line = train_dataloader.load_custom("corpus")
        # concat title and text
        train_corpus = dict(map(
            lambda k: (k, f"{train_corpus[k]['title']}. {train_corpus[k]['text']}"),
            train_corpus
        ))
        print(f"Loaded {len(train_corpus)} documents")

        # tokenize
        train_corpus = dict(map(
            lambda k: (
                k,
                " ".join(
                    map(
                        lambda i: str(i),
                        tokenizer(
                            k,
                            padding=True,
                            truncation=True,
                            return_tensors="pt",
                            max_length=350,
                        )["input_ids"][:, 1:-1].tolist()[0]
                    )
                )
            ),
            tqdm(train_corpus,
                 desc="Tokenizing Documents",
                 unit_scale=1000000)
        ))
        pipe = pipe.fit(train_corpus.values())
        d_idf = pipe['tfidf'].idf_
        d_idf[d_idf == inf] = 0
        d_idf = d_idf.tolist()

        # ============ Queries ============ #
        pipe = Pipeline([
            ('count', CountVectorizer(vocabulary=vocab_ids)),
            ('tfidf', TfidfTransformer(smooth_idf=False))
        ])
        # load
        train_queries = train_dataloader.load_custom("queries")
        # concat title and text
        print(f"Loaded {len(train_queries)} Queries")

        # tokenize
        train_queries = dict(map(
            lambda k: (
                k,
                " ".join(
                    map(
                        lambda i: str(i),
                        tokenizer(
                            k,
                            padding=True,
                            truncation=True,
                            return_tensors="pt",
                            max_length=350,
                        )["input_ids"][:, 1:-1].tolist()[0]
                    )
                )
            ),
            tqdm(train_queries,
                 desc="Tokenizing Queries",
                 unit_scale=1000000)
        ))
        pipe = pipe.fit(train_queries.values())
        q_idf = pipe['tfidf'].idf_
        q_idf[q_idf == inf] = 0
        q_idf = q_idf.tolist()

        # ============ Save ============ #
        with (dataset_dir / "idf" / "d_idf.json").open("w") as fp:
            json.dump(d_idf, fp)

        with (dataset_dir / "idf" / "q_idf.json").open("w") as fp:
            json.dump(q_idf, fp)


