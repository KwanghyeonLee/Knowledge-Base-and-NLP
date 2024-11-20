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
        "robust04"
        # "arguana",
        # "climate-fever",
        # # "cqadupstack",
        # "dbpedia-entity",
        # "fever",
        # "fiqa",
        # "hotpotqa",
        # # "msmarco-v2",
        # "msmarco",
        # "nfcorpus",
        # "nq",
        # "quora",
        # "scidocs",
        # "scifact",
        # "trec-covid",
        # "webis-touche2020",
    ]
    for data_name in data_names:
        print(f"Processing {data_name}")
        model_name = "naver/splade-cocondenser-ensembledistil"
        dtype = "beir"

        root_dir = Path("/workspace")
        type_dir = root_dir / dtype
        data_dir = type_dir / data_name

        # if dtype == "gpl":
        #     GPLDownloader(data_dir=str(type_dir.absolute()), dataset_name=data_name)
        # elif dtype == "beir":
        #     BEIRDownloader(data_dir=str(type_dir.absolute()), dataset_name=data_name)

        tf_dir = data_dir / "tf"
        tf_dir.mkdir(exist_ok=True)

        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

        vocab_dict = sorted(tokenizer.vocab.items(), key=lambda x: x[1])
        vocab_dict_rev = dict(map(lambda i: (i[1], i[0]), vocab_dict))
        vocab_tokens = list(map(lambda i: i[0], vocab_dict))
        vocab_ids = list(map(lambda i: str(i[1]), vocab_dict))

        pipe = Pipeline([
            ('count', CountVectorizer(vocabulary=vocab_ids)),
            # ('tfidf', TfidfTransformer())
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
        tf = pipe['count'].transform(train_corpus.values()).sum(axis=0)
        tf = tf.astype(int)
        log_tf = np.log10(tf)

        tf = tf.tolist()[0]
        log_tf = log_tf.tolist()[0]

        with (tf_dir / "tfs.json").open("w") as fp:
            json.dump(tf, fp)

        with (tf_dir / "top100tfs.json").open("w") as fp:
            json.dump(sorted(zip(tf, vocab_tokens), key=lambda i: i[0], reverse=True)[:100], fp)

        with (tf_dir / "id2token.json").open("w") as fp:
            json.dump(vocab_dict_rev, fp)

        plt.plot(tf)
        plt.title(f"TF {data_name}")
        plt.ylabel("TF")
        plt.xlabel("Token ID")
        # plt.show()
        plt.savefig(tf_dir / "tf_order.png")
        plt.close()

        plt.plot(sorted(log_tf, reverse=True))
        plt.title(f"TF {data_name} (sorted)")
        plt.ylabel("TF (log)")
        # plt.yscale("log")
        # plt.show()
        plt.savefig(tf_dir / "tf_sorted.png")
        plt.close()
