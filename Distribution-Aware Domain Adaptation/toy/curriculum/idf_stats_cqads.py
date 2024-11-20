import json

from pathlib import Path
import gensim
import numpy as np
import requests
import transformers
from numpy import inf
from tqdm.rich import tqdm

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

from matplotlib import pyplot as plt

from src.datamodule.beir.downloader import BEIRDownloader
from src.datamodule.gpl.downloader import GPLDownloader
from src.datamodule.modules.dataloader import GenericDataLoader


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

if __name__ == '__main__':
    # data_name = "trec-covid-v2"
    data_names = [
        # "arguana",
        # "climate-fever",
        # "bioasq",
        # "signal-1m",
        # "trec-news",
        # "robust04",
        # "webis-touche2020",
        # "dbpedia-entity",
        # "fever",
        # "fiqa",
        # "hotpotqa",
        # "nfcorpus",
        # "nq",
        # "quora",
        # "scidocs",
        # "scifact",
        # "trec-covid",
        "cqadupstack/android",
        "cqadupstack/english",
        "cqadupstack/gaming",
        "cqadupstack/gis",
        "cqadupstack/mathematica",
        "cqadupstack/physics",
        "cqadupstack/programmers",
        "cqadupstack/stats",
        "cqadupstack/tex",
        "cqadupstack/unix",
        "cqadupstack/webmasters",
        "cqadupstack/wordpress",
    ]

    train_corpus = {}

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


        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

        vocab_dict = sorted(tokenizer.vocab.items(), key=lambda x: x[1])
        vocab_dict_rev = dict(map(lambda i: (i[1], i[0]), vocab_dict))
        vocab_tokens = list(map(lambda i: i[0], vocab_dict))
        vocab_ids = list(map(lambda i: str(i[1]), vocab_dict))

        pipe = Pipeline([
            ('count', CountVectorizer(vocabulary=vocab_ids)),
            ('tfidf', TfidfTransformer())
        ])

        # loader
        train_dataloader = GenericDataLoader(
            data_folder=str(data_dir.absolute()),
            prefix="qgen",
        )
        # load
        sub_train_corpus = train_dataloader.load_custom("corpus")

        # prepend name to key
        name = data_name.split("/")[-1]
        sub_train_corpus = dict(map(
            lambda k: (f"{name}_{k}", sub_train_corpus[k]),
            sub_train_corpus
        ))

        # concat title and text
        sub_train_corpus = dict(map(
            lambda k: (k, f"{sub_train_corpus[k]['title']}. {sub_train_corpus[k]['text']}"),
            sub_train_corpus
        ))
        train_corpus.update(sub_train_corpus)

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
             desc="Tokenizing",
             unit_scale=1000000)
    ))
    pipe = pipe.fit(train_corpus.values())
    idf = pipe['tfidf'].idf_.tolist()

    for data_name in data_names:
        dtype = "beir"

        root_dir = Path("/workspace")
        type_dir = root_dir / dtype
        data_dir = type_dir / data_name

        idf_dir = data_dir / "idf"
        idf_dir.mkdir(exist_ok=True)

        with (idf_dir / "idfs.json").open("w") as fp:
            json.dump(idf, fp)

        requests.post('https://hooks.slack.com/services/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',
                      json={'text': f'{data_name} Done'})



