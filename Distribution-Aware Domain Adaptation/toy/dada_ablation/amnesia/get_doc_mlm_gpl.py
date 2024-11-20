import json
from pathlib import Path

import hydra
import torch
import typer
from omegaconf import DictConfig
from torch import tensor
from tqdm.rich import tqdm
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer

from src.datamodule.modules.dataloader import GenericDataLoader
from src.models.trainer import RetrievalModel, HybridRetrievalModel
from src.predict import get_prediction_model


def amnesia(
        dataname: str = "robust04",
        cuda_id: int = 2,
):
    device = torch.device(f'cuda:{cuda_id}')
    model_name = f"GPL/{dataname}-msmarco-distilbert-gpl"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model = model.eval()
    model.to(device)

    mlm_head = AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased").vocab_projector
    mlm_head = mlm_head.eval()
    mlm_head.to(device)

    root_dir = Path("/workspace")
    beir_dir = root_dir / "beir"

    amnesia_dir = root_dir / "amnesia"
    expr_name = "gpl-base"
    target_dir = amnesia_dir / expr_name / dataname
    target_dir.mkdir(exist_ok=True, parents=True)
    mlm_pth = target_dir / "mlm.jsonl"

    batch_size = 256

    # loader
    (corpus, _) = GenericDataLoader(
        data_folder=str((beir_dir / dataname).absolute()),
    ).load_custom(which="corpus", split="test")

    corpus = list(corpus.items())
    for k in tqdm(range(0, len(corpus) // batch_size),
                  desc=f"MLM: {dataname}",
                  unit_scale=1000000):
        batch_corpus = []
        batch_ids = []
        batch_idxs = range(k * batch_size, (k + 1) * batch_size)
        for i in batch_idxs:
            doc = corpus[i]
            batch_ids.append(doc[0])
            batch_corpus.append(
                f'{doc[1]["title"]}. {doc[1]["text"]}'
            )

        with torch.no_grad():
            docs_tokenized = tokenizer(
                batch_corpus,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            docs_tokenized = docs_tokenized.to(device)

            docs_embed = model(**docs_tokenized).last_hidden_state.mean(dim=1)
            docs_mlm = mlm_head(docs_embed)

        with mlm_pth.open("a") as f:
            for did, doc_mlm in zip(batch_ids, docs_mlm):
                doc_mlm = doc_mlm.tolist()
                doc_mlm = json.dumps(
                    {
                        "id": did,
                        "mlm": doc_mlm
                    }
                )
                f.write(doc_mlm)
                f.write("\n")

    import requests
    requests.post(
        'https://hooks.slack.com/services/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',
        json={'text': f'{dataname} mlm Done'}
    )

if __name__ == '__main__':
    data_names = [
        # "nfcorpus",
        # "scidocs",
        # "scifact",
        "fiqa",
        # "robust04",
    ]
    for dataname in data_names:
        amnesia(dataname)
