import json
import random
from pathlib import Path

import nltk
from pyserini.search import LuceneSearcher
from tqdm.rich import tqdm

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
        # "nfcorpus",
        # "nq",
        # "quora",
        # "scidocs",
        # "scifact",
        "trec-covid",
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

        exp_dir = Path("/workspace/MDSQ/experiments/mdpq") / data_name
        exp_dir.mkdir(exist_ok=True, parents=True)

        # loader
        train_dataloader = GenericDataLoader(
            data_folder=str(data_dir.absolute()),
        )
        # load
        train_corpus, train_corpus_id2_line = train_dataloader.load_custom("corpus")

        # concat title and text
        train_corpus = dict(map(
            lambda k: (k, f"{train_corpus[k]['title']}. {train_corpus[k]['text']}"),
            train_corpus
        ))
        print(f"Loaded {len(train_corpus)} documents")

        searcher = LuceneSearcher.from_prebuilt_index(f'beir-v1.0.0-{data_name}.flat')

        if not (exp_dir / "queries.jsonl").exists():
            with (exp_dir / "queries.jsonl").open("w") as qf:
                for docid in tqdm(train_corpus):
                    sentences = nltk.sent_tokenize(train_corpus[docid])

                    # random select 5 sentences
                    sentences = random.choices(sentences, k=5)

                    for sent_idx, sentence in enumerate(sentences):
                        query = sentence.strip()
                        qf.write(json.dumps({
                            "id": f"{docid}-sent{sent_idx}",
                            "text": query
                        }) + "\n")

        # if not (exp_dir / "qrels.tsv").exists():
        with (exp_dir / "queries.jsonl").open("r") as qf:
            n_lines = len(qf.readlines())
        if (exp_dir / "qrels.tsv").exists():
            (exp_dir / "qrels.tsv").unlink()

        print(f"Generating qrels.tsv for {n_lines} queries")

        with (exp_dir / "queries.jsonl").open("r") as qf:
            with (exp_dir / "qrels.tsv").open("w") as qrelf:
                qrelf.write("qid\tdid\tlabel\n")
                for row_text in tqdm(qf, total=n_lines):
                    row = json.loads(row_text)
                    qid, query = row["id"], row["text"]

                    origin_did = qid.split("-sent")[0]

                    hits = searcher.search(query, k=100)
                    results = {hit.docid: hit.score for hit in hits}
                    sorted_results = sorted(results.items(), key=lambda item: item[1], reverse=True)

                    for did, score in sorted_results:
                        qrelf.write(f"{qid}\t{did}\t{score}\n")

        print()