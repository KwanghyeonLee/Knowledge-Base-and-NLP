import json
from functools import reduce

from pathlib import Path
import gensim
import numpy as np
import torch
import transformers
import typer
from einops import repeat, rearrange
from numpy import inf
from torch import log, relu, tensor, nn
from tqdm.rich import tqdm

from src.datamodule.gpl.datamodule import GenerativePseudoLabelingDataset
from src.datamodule.gpl.downloader import GPLDownloader
from src.datamodule.modules.dataloader import GenericDataLoader
from src.losses.modules.js_div import JSDivergence
from src.losses.modules.kl_div import KLDivergenceLoss

def main(
        data_name: str,
        dtype: str = None,
        idf_name: str = "idfs.json",
        cuda_id: int = 0,
        batch_size: int = 64,
):
    # data_name = "trec-covid-v2"
    if dtype is None:
        raise ValueError("dtype must be specified")

    model_name = "naver/splade_v2_distil"
    print(f"Using {model_name} for tokenization and embedding")

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModelForMaskedLM.from_pretrained(model_name)
    model = model.eval()
    model = model.to(f"cuda:{cuda_id}")
    output_org = lambda out: log(1 + relu(out)).amax(dim=1)

    obj = JSDivergence()

    # data_names = [
    #     # "arguana",
    #     # "climate-fever",
    #     # "cqadupstack",
    #     # "dbpedia-entity",
    #     # "fever",
    #     # "fiqa",
    #     # "hotpotqa",
    #     # "msmarco-v2",
    #     # "msmarco",
    #     "nfcorpus",
    #     # "nq",
    #     # "quora",
    #     # "scidocs",
    #     "scifact",
    #     "trec-covid",
    #     # "webis-touche2020",
    # ]

    print(f"Processing {data_name}")

    root_dir = Path("/workspace")
    target_dir = root_dir / dtype

    # loader
    (corpus, _), queries, qrels = GenericDataLoader(
        data_folder=str((target_dir / data_name).absolute()),
        prefix="qgen",
    ).load(split="train")

    target_dataset = GenerativePseudoLabelingDataset(
        data_dir=(target_dir / data_name).absolute(),
        queries=queries,
        corpus=corpus,
        qrels=qrels,
    )
    # idf
    idfs = tensor(
        json.load(
            (target_dir / data_name / "idf" / idf_name).open("r")
        ),
        device=f"cuda:{cuda_id}",
    )
    masked_idf = idfs.clone()
    masked_idf[masked_idf == 0] = -inf

    # clear directory
    if (target_dir / data_name / "curriculum").is_dir():
        for pt in (target_dir / data_name / "curriculum").iterdir():
            pt.unlink()
        (target_dir / data_name / "curriculum").rmdir()

    (target_dir / data_name / "curriculum").mkdir(exist_ok=True, parents=True)

    for k in tqdm(range(0, len(target_dataset) // batch_size),
                  desc="Tokenizing & Embedding",
                  unit_scale=1000000):
        batch_corpus = []
        batch_ids = range(k * batch_size, (k + 1) * batch_size)
        for i in batch_ids:
            batch_corpus.append(target_dataset[i])
        batch_corpus = [row.doc for row in batch_corpus]
        batch_corpus = reduce(lambda i, j: i + j, batch_corpus)

        with torch.no_grad():
            docs_tokenized = tokenizer(
                batch_corpus,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                max_length=350,
            )
            docs_tokenized = docs_tokenized.to(f"cuda:{cuda_id}")
            splade_score = output_org(model(**docs_tokenized)["logits"])
            masked_splade_score = splade_score.clone()
            masked_splade_score[masked_splade_score == 0] = -inf
            # loss = obj(
            #     splade_score.log_softmax(dim=-1),
            #     repeat(idfs, "d -> b d", b=len(splade_score)).softmax(dim=-1)
            # ).mean(dim=-1)
            # loss = obj(
            #     splade_score,
            #     repeat(idfs, "d -> b d", b=len(splade_score))
            # ).mean(dim=-1)
            loss = obj(
                masked_splade_score,
                repeat(masked_idf, "d -> b d", b=len(splade_score))
            ).mean(dim=-1)
            loss = rearrange(loss, "(b d) -> b d", d=2).mean(dim=-1)
            idxs = tensor(batch_ids, device=f"cuda:{cuda_id}", dtype=torch.long)

            save_target = torch.cat([idxs.unsqueeze(-1), loss.unsqueeze(-1)], dim=-1)
            torch.save(save_target, target_dir / data_name / "curriculum" / f"{k}.pt")

if __name__ == '__main__':
    typer.run(main)
    # main("robust04", "gpl", "idfs_zero_unseen.json", 0)

