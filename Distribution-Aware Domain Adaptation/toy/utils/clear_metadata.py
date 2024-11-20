import json
from pathlib import Path

from tqdm.rich import tqdm

from src.datamodule.modules.dataloader import GenericDataLoader

root_dir = Path("/workspace")

def clear_query(
        data_name: str,
):
    data_dir = root_dir / "beir" / data_name

    if data_name == "cqadupstack":
        subsets = list(map(
            lambda pth: pth.parts[-1],
            data_dir.glob("*")
        ))

    else:
        subsets = [""]


    if "idf" in subsets:
        subsets.pop(subsets.index("idf"))

    for subset in subsets:
        print(f"Processing {data_name}/{subset}...")

        queries = GenericDataLoader(
            data_folder=str((data_dir / subset).absolute()),
        ).load_custom("queries")

        # drop metadata the query
        with (data_dir / subset / "queries.jsonl.shorten").open("w") as f:
            for k, v in tqdm(queries.items(), desc="Shortening queries"):
                data = {
                    "_id": k,
                    "text": v,
                    "metadata": {}
                }

                f.write(json.dumps(data) + '\n')

        print("Removing queries.jsonl...")
        (data_dir / subset / "queries.jsonl").unlink()

        print("Renaming queries.jsonl.shorten to queries.jsonl...")
        (data_dir / subset / "queries.jsonl.shorten").rename(
            data_dir / subset / "queries.jsonl"
        )

def clear_doc(
        dtype: str,
        data_name: str,
):
    data_dir = root_dir / dtype / data_name

    if data_name == "cqadupstack":
        subsets = list(map(
            lambda pth: pth.parts[-1],
            data_dir.glob("*")
        ))

    else:
        subsets = [""]


    if "idf" in subsets:
        subsets.pop(subsets.index("idf"))

    for subset in subsets:
        print(f"Processing {data_name}/{subset}...")

        corpus = GenericDataLoader(
            data_folder=str((data_dir / subset).absolute()),
        ).load_custom("corpus")

        # drop metadata the corpus
        with (data_dir / subset / "corpus.jsonl.shorten").open("w") as f:
            for k, v in tqdm(corpus.items(), desc="Shortening corpus"):
                data = {
                    "_id": k,
                    "title": v["title"],
                    "text": v["text"],
                    "metadata": {}
                }

                f.write(json.dumps(data) + '\n')

        print("Removing corpus.jsonl...")
        (data_dir / subset / "corpus.jsonl").unlink()

        print("Renaming corpus.jsonl.shorten to corpus.jsonl...")
        (data_dir / subset / "corpus.jsonl.shorten").rename(
            data_dir / subset / "corpus.jsonl"
        )


if __name__ == '__main__':
    clear_doc("beir", "bioasq")
