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
        # "climate-fever",
        # "cqadupstack",
        # "dbpedia-entity",
        # "fever",
        # "fiqa",
        # "hotpotqa",
        # "msmarco-v2",
        # "msmarco",
        # "nfcorpus",
        # "nq",
        # "quora",
        # "scidocs",
        # "scifact",
        # "trec-covid",
        "webis-touche2020",
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
            downloader.download(data_name)
        elif dtype == "beir":
            downloader = BEIRDownloader(data_dir=str(type_dir.absolute()))
            downloader.download(data_name)
        else:
            pass

        idf_dir = data_dir / "idf"
        idf_dir.mkdir(exist_ok=True)

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
        train_corpus, _ = train_dataloader.load_custom("corpus")
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
        idf = pipe['tfidf'].idf_
        n = len(train_corpus)
        df = ( (1 + n) / np.exp(idf - 1) ) - 1
        df = df.astype(int)
        log_df = np.log10(df)

        df = df.tolist()
        log_df = log_df.tolist()

        with (idf_dir / "dfs.json").open("w") as fp:
            json.dump(df, fp)

        with (idf_dir / "top100dfs.json").open("w") as fp:
            json.dump(sorted(zip(df, vocab_tokens), key=lambda i: i[0], reverse=True)[:100], fp)

        with (idf_dir / "id2token.json").open("w") as fp:
            json.dump(vocab_dict_rev, fp)

        plt.plot(df)
        plt.title(f"DF {data_name}")
        plt.ylabel("DF")
        plt.xlabel("Token ID")
        # plt.show()
        plt.savefig(idf_dir / "idf_order.png")
        plt.close()

        plt.plot(sorted(log_df, reverse=True))
        plt.title(f"DF {data_name} (sorted)")
        plt.ylabel("DF (log)")
        # plt.yscale("log")
        # plt.show()
        plt.savefig(idf_dir / "idf_sorted.png")
        plt.close()
