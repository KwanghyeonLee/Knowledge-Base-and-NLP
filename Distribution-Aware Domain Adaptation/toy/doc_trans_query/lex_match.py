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
        model_name = "Luyu/co-condenser-marco"
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
                tokenizer(
                    k,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=512,
                )["input_ids"][0, ]
            ),
            tqdm(train_corpus,
                 desc="Tokenizing",
                 unit_scale=1000000)
        ))
        train_query = dict(map(
            lambda k: (
                k,
                tokenizer(
                    k,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=512,
                )["input_ids"][0, ]
            ),
            tqdm(train_query,
                 desc="Tokenizing",
                 unit_scale=1000000)
        ))

        model = transformers.AutoModel.from_pretrained(model_name)
        model.to("cuda:0")
        model.eval()
        # sims = defaultdict(dict)
        # not_sims = defaultdict(dict)

        sim = 0
        sim_cnt = 0
        not_sim = 0
        not_sim_cnt = 0

        with torch.no_grad():
            combs = list(product(train_query.keys(), train_corpus.keys()))
            for idx, (qid, did) in tqdm(list(enumerate(combs)), desc="Measuring LEX-SEM overlap"):
                q_tok = train_query[qid].to("cuda:0")
                q_emb = model(q_tok.unsqueeze(0)).last_hidden_state[0]

                d_tok = train_corpus[did].to("cuda:0")
                d_emb = model(d_tok.unsqueeze(0)).last_hidden_state[0]

                lex_overlap = torch.zeros((len(q_tok), len(d_tok)), device="cuda:0")
                for i, j in product(range(1, len(q_tok)-1), range(1, len(d_tok)-1)):
                    if q_tok[i] == d_tok[j]:
                        lex_overlap[i, j] = 1

                sem_sim = q_emb @ d_emb.T

                sem_sim_mask = 1 * (sem_sim > sem_sim.mean())

                masked_sem_sim = sem_sim_mask * lex_overlap

                lex_sem_overlap = torch.count_nonzero(masked_sem_sim).item()

                score = 0
                if qid in train_qrels and did in train_qrels[qid]:
                    if int(train_qrels[qid][did]) != 0:
                        score = int(train_qrels[qid][did])

                if score > 0:
                    sim += lex_sem_overlap
                    sim_cnt += 1
                else:
                    not_sim += lex_sem_overlap
                    not_sim_cnt += 1

                if idx % 100_000 == 0 and idx > 0:
                    if sim_cnt > 0:
                        print(f"Sim: {sim / sim_cnt}")
                    if not_sim_cnt > 0:
                        print(f"Not Sim: {not_sim / not_sim_cnt}")

        print(f"Sim: {sim / sim_cnt}")
        print(f"Not Sim: {not_sim / not_sim_cnt}")


        # with (exp_dir / "sims.json").open("w") as fp:
        #     json.dump(sims, fp)
        #
        # with (exp_dir / "not_sims.json").open("w") as fp:
        #     json.dump(not_sims, fp)


        print()

