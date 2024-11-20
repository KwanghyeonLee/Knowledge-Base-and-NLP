import json
from pathlib import Path

from tqdm.rich import tqdm

from src.datamodule.modules.dataloader import GenericDataLoader


def gpl_filter(
        name: str,
):
    root_dir = Path("/workspace")
    tmp_dir = root_dir / "tmp"
    gpl_dir = root_dir / "gpl" / name

    gpl_corpus_list = (gpl_dir / "corpus.doc_ids.txt").open("r").readlines()
    gpl_corpus_list = list(map(lambda x: x.strip(), gpl_corpus_list))

    corpus = GenericDataLoader(
        data_folder=str(gpl_dir.absolute()),
    ).load_custom(which="corpus", split="test")

    # drop metadata the query
    with (gpl_dir / "corpus.jsonl.shorten").open("w") as f:
        for gpl_corpus_id in tqdm(gpl_corpus_list, desc="reducing corpus"):
            k = gpl_corpus_id
            v = corpus[k]
            data = {
                "_id": k,
                "title": v["title"],
                "text": v["text"],
                "metadata": {}
            }

            f.write(json.dumps(data) + '\n')

    print("Removing corpus.jsonl...")
    (gpl_dir / "corpus.jsonl").unlink()

    print("Renaming queries.jsonl.shorten to queries.jsonl...")
    (gpl_dir / "corpus.jsonl.shorten").rename(
        gpl_dir / "corpus.jsonl"
    )


    print(f"Processing {name}")




if __name__ == '__main__':
    # gpl_filter("robust04")
    gpl_filter("bioasq")