import json
from collections import defaultdict
from itertools import product

from pathlib import Path
import gensim
import numpy as np
import torch
import transformers
from numpy import inf
from torch import tensor
from tqdm.rich import tqdm

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

from matplotlib import pyplot as plt

from src.datamodule.beir.downloader import BEIRDownloader
from src.datamodule.gpl.downloader import GPLDownloader
from src.datamodule.modules.dataloader import GenericDataLoader
from src.losses.modules.kl_div import KLDivergenceLoss


def sent_to_words(sentences, tokenizer):
    for sentence in tqdm(sentences):
        # deacc=True removes punctuations
        gen_pre = " ".join(gensim.utils.simple_preprocess(str(sentence), deacc=True))
        yield tokenizer(
            gen_pre,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=350,
        )

kl_div = KLDivergenceLoss()
def measure_similarity(idf_a, idf_b):
    # sim = kl_div(
    #     tensor(idf_a),
    #     tensor(idf_b)
    # )
    if type(idf_a) == list:
        idf_a = tensor(idf_a)
    if type(idf_b) == list:
        idf_b = tensor(idf_b)
    P = idf_a
    Q = idf_b
    M = 0.5 * (P + Q)
    sim = 0.5 * (kl_div(P, M) + kl_div(Q, M))
    return sim


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
        # "nfcorpus",
        # "nq",
        # "quora",
        # "scidocs",
        "scifact",
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

        downloader = None
        if dtype == "gpl":
            downloader = GPLDownloader(data_dir=str(type_dir.absolute()))
        elif dtype == "beir":
            downloader = BEIRDownloader(data_dir=str(type_dir.absolute()))

        downloader.download(data_name)

        exp_dir = Path("/workspace/experiments") / data_name
        exp_dir.mkdir(exist_ok=True, parents=True)

        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

        vocab_dict = sorted(tokenizer.vocab.items(), key=lambda x: x[1])
        vocab_dict_rev = dict(map(lambda i: (i[1], i[0]), vocab_dict))
        vocab_tokens = list(map(lambda i: i[0], vocab_dict))
        vocab_ids = list(map(lambda i: str(i[1]), vocab_dict))

        # loader
        train_dataloader = GenericDataLoader(
            data_folder=str(data_dir.absolute()),
            # prefix="qgen",
        )
        # load
        train_corpus, train_corpus_id2_line = train_dataloader.load_custom("corpus")
        train_query = train_dataloader.load_custom("queries")
        train_qrels = train_dataloader.load_custom("qrels", "test")
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
                            max_length=512,
                        )["input_ids"][:, 1:-1].tolist()[0]
                    )
                )
            ),
            tqdm(train_corpus,
                 desc="Tokenizing",
                 unit_scale=1000000)
        ))
        train_query = dict(map(
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
                            max_length=512,
                        )["input_ids"][:, 1:-1].tolist()[0]
                    )
                )
            ),
            tqdm(train_query,
                 desc="Tokenizing",
                 unit_scale=1000000)
        ))
        corpus_pipe = Pipeline([
            ('count', CountVectorizer(vocabulary=vocab_ids)),
            ('tfidf', TfidfTransformer())
        ])
        query_pipe = Pipeline([
            ('count', CountVectorizer(vocabulary=vocab_ids)),
            ('tfidf', TfidfTransformer())
        ])

        corpus_pipe = corpus_pipe.fit(train_corpus.values())
        query_pipe = query_pipe.fit(train_query.values())


        # sims = defaultdict(dict)
        # not_sims = defaultdict(dict)

        sim = 0
        sim_cnt = 0
        not_sim = 0
        not_sim_cnt = 0

        combs = list(product(train_query.keys(), train_corpus.keys()))
        for idx, (qid, did) in tqdm(list(enumerate(combs)), desc="Computing TF"):
            doc_tfidf = corpus_pipe["count"].transform([train_corpus[did]])
            query_tfidf = query_pipe["count"].transform([train_query[qid]])

            doc_tfidf = torch.from_numpy(doc_tfidf.toarray()[0])
            query_tfidf = torch.from_numpy(query_tfidf.toarray()[0])

            doc_tfidf = doc_tfidf.float()
            query_tfidf = query_tfidf.float()

            sim = measure_similarity(doc_tfidf, query_tfidf)
            if qid in train_qrels and did in train_qrels[qid]:
                if int(train_qrels[qid][did]) == 0:
                    # not_sims[qid][did] = sim.item()
                    not_sim += sim.item()
                    not_sim_cnt += 1
                else:
                    # sims[qid][did] = sim.item()
                    sim += sim.item()
                    sim_cnt += 1
            else:
                # not_sims[qid][did] = sim.item()
                not_sim += sim.item()
                not_sim_cnt += 1


            if idx % 100_000 == 0:
                print(f"Sim: {sim / sim_cnt}")
                print(f"Not Sim: {not_sim / not_sim_cnt}")

        print(f"Sim: {sim / sim_cnt}")
        print(f"Not Sim: {not_sim / not_sim_cnt}")

        # with (exp_dir / "sims.json").open("w") as fp:
        #     json.dump(sims, fp)
        #
        # with (exp_dir / "not_sims.json").open("w") as fp:
        #     json.dump(not_sims, fp)


        print()

