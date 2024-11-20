import json

from pathlib import Path
import gensim
import numpy as np
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
        # "cqadupstack",
        # "dbpedia-entity",
        # "fever",
        # "robust04",
        "fiqa",
        # "hotpotqa",
        # "msmarco-v2",
        # "msmarco",
        # "nfcorpus",
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
        model_name = f"GPL/{data_name}-msmarco-distilbert-gpl"
        dtype = "gpl"

        root_dir = Path("/workspace")
        beir_dir = root_dir / "beir"
        data_dir = beir_dir / data_name

        amnesia_dir = root_dir / "amnesia"
        expr_name = "gpl-base"
        target_dir = amnesia_dir / expr_name / data_name
        target_dir.mkdir(exist_ok=True, parents=True)

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
        )
        # load
        train_corpus, _ = train_dataloader.load_custom(which="corpus", split="test")
        # concat title and text
        train_corpus = dict(map(
            lambda k: (k, f"{train_corpus[k]['title']}. {train_corpus[k]['text']}"),
            train_corpus
        ))
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

        with (target_dir / "idf.json").open("w") as fp:
            json.dump(idf, fp)

