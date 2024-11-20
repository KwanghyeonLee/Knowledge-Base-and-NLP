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
        "arguana",
        "climate-fever",
        # "cqadupstack",
        "dbpedia-entity",
        "fever",
        "fiqa",
        "hotpotqa",
        # "msmarco-v2",
        "msmarco",
        "nfcorpus",
        "nq",
        "quora",
        "scidocs",
        "scifact",
        "trec-covid",
        "webis-touche2020",
    ]
    for data_name in data_names:
        print(f"Processing {data_name}")
        model_name = "naver/splade-cocondenser-ensembledistil"
        dtype = "beir"

        root_dir = Path("/workspace")
        type_dir = root_dir / dtype
        data_dir = type_dir / data_name

        if dtype == "gpl":
            GPLDownloader(data_dir=str(type_dir.absolute()), dataset_name=data_name)
        elif dtype == "beir":
            BEIRDownloader(data_dir=str(type_dir.absolute()), dataset_name=data_name)

        dfitf_dir = data_dir / "dfitf"
        dfitf_dir.mkdir(exist_ok=True)

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
            dtype=dtype,
            data_folder=str(data_dir.absolute()),
            prefix="qgen",
        )
        # load
        train_corpus = train_dataloader.load_custom("corpus")
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

        n = len(train_corpus)
        # df
        idf = pipe['tfidf'].idf_
        df = ((1 + n) / np.exp(idf - 1)) - 1

        summed_dfitf = np.zeros(len(vocab_tokens))

        for doc in tqdm(
            train_corpus.values(),
            desc="DF-ITF calculating...",
            unit_scale=1000000
        ):
            # itf
            tf = pipe['count'].transform([doc]).toarray()[0]
            itf = np.log( (n + 1) / (tf + 1) ) + 1
            summed_dfitf += df * itf

        # dfitf
        # summed_dfitf = (df * itf).sum(axis=0)
        log_summed_dfitf = np.log10(summed_dfitf)

        summed_dfitf = summed_dfitf.tolist()
        log_summed_dfitf = log_summed_dfitf.tolist()

        with (dfitf_dir / "dfitf.json").open("w") as fp:
            json.dump(summed_dfitf, fp)

        with (dfitf_dir / "top100dfitfs.json").open("w") as fp:
            json.dump(sorted(zip(summed_dfitf, vocab_tokens), key=lambda i: i[0], reverse=True)[:100], fp)

        with (dfitf_dir / "id2token.json").open("w") as fp:
            json.dump(vocab_dict_rev, fp)

        plt.plot(summed_dfitf)
        plt.title(f"DF-ITF {data_name}")
        plt.ylabel("DF-ITF")
        plt.xlabel("Token ID")
        # plt.show()
        plt.savefig(dfitf_dir / "dfitf_order.png")
        plt.close()

        plt.plot(sorted(log_summed_dfitf, reverse=True))
        plt.title(f"DF-ITF {data_name} (sorted)")
        plt.ylabel("DF-ITF (log)")
        # plt.yscale("log")
        # plt.show()
        plt.savefig(dfitf_dir / "dfitf_sorted.png")
        plt.close()
