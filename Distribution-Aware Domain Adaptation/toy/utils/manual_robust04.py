import json
from pathlib import Path

import ir_datasets
from tqdm import tqdm

root_dir = Path("/workspace")
tmp_dir = root_dir / "tmp"
type_dir = root_dir / "beir"
data_dir = type_dir / "robust04"

dataset = ir_datasets.load("disks45/nocr")
with (data_dir / "corpus.jsonl").open("w", encoding="utf-8") as fp:
    for doc in tqdm(dataset.docs_iter(), desc='Download robust04 (doc)'):
        data = {
            "_id": doc.doc_id,
            "text": doc.body,
            "title": doc.title,
            "metadata": {}
        }

        fp.write(json.dumps(data) + '\n')



