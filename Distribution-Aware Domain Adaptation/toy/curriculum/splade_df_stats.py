import json

from pathlib import Path
import gensim
import numpy as np
import torch
import transformers
from numpy import inf
from torch import relu, log
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
        with torch.no_grad():
            print(f"Processing {data_name}")
            # 'Luyu/co-condenser-marco'
            model_name = "naver/splade-cocondenser-ensembledistil"
            dtype = "beir"

            root_dir = Path("/workspace")
            type_dir = root_dir / dtype
            data_dir = type_dir / data_name

            if dtype == "gpl":
                GPLDownloader(data_dir=str(type_dir.absolute()), dataset_name=data_name)
            elif dtype == "beir":
                BEIRDownloader(data_dir=str(type_dir.absolute()), dataset_name=data_name)

            splade_f_dir = data_dir / "splade-freq"
            splade_f_dir.mkdir(exist_ok=True)

            tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
            model = transformers.AutoModelForMaskedLM.from_pretrained(model_name).to("cuda:0")
            model.eval()

            vocab_dict = sorted(tokenizer.vocab.items(), key=lambda x: x[1])
            vocab_dict_rev = dict(map(lambda i: (i[1], i[0]), vocab_dict))
            vocab_tokens = list(map(lambda i: i[0], vocab_dict))
            vocab_ids = list(map(lambda i: str(i[1]), vocab_dict))

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
            # tokenize & model output
            keys = list(train_corpus.keys())
            batch_size = 64
            splade_f = torch.ones(len(vocab_tokens)).to("cuda:0")
            for k in tqdm(range(0, len(keys) // batch_size + 1),
                          desc="Tokenizing & Embedding",
                          unit_scale=1000000):
                batch_keys = keys[k * batch_size: (k + 1) * batch_size]
                batch_corpus = list(map(lambda i: train_corpus[i], batch_keys))
                toks = tokenizer(
                        batch_corpus,
                        padding=True,
                        truncation=True,
                        return_tensors="pt",
                        max_length=350,
                    )["input_ids"].to("cuda:0")
                output = model(input_ids=toks, output_hidden_states=True)
                splade_sc = log(1 + relu(output.logits)).sum(dim=1).sum(dim=0).squeeze()
                splade_f += splade_sc

            splade_f = splade_f.cpu().numpy()
            log_splade_f = np.log10(splade_f)

            splade_f = splade_f.tolist()
            log_splade_f = log_splade_f.tolist()

            with (splade_f_dir / "splade_fs.json").open("w") as fp:
                json.dump(splade_f, fp)

            with (splade_f_dir / "top100splade_fs.json").open("w") as fp:
                json.dump(sorted(zip(splade_f, vocab_tokens), key=lambda i: i[0], reverse=True)[:100], fp)

            with (splade_f_dir / "id2token.json").open("w") as fp:
                json.dump(vocab_dict_rev, fp)

            plt.plot(splade_f)
            plt.title(f"SPLADE FREQ {data_name}")
            plt.ylabel("SPLADE FREQ")
            plt.xlabel("Token ID")
            # plt.show()
            plt.savefig(splade_f_dir / "splade_f_order.png")
            plt.close()

            plt.plot(sorted(log_splade_f, reverse=True))
            plt.title(f"SPLADE FREQ {data_name} (sorted)")
            plt.ylabel("SPLADE FREQ (log)")
            # plt.yscale("log")
            # plt.show()
            plt.savefig(splade_f_dir / "splade_f_sorted.png")
            plt.close()
