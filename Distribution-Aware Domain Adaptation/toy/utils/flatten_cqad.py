from pathlib import Path

from src.datamodule.modules.dataloader import GenericDataLoader

root_dir = Path("/workspace")
tmp_dir = root_dir / "tmp"
cqad_dir = root_dir / "beir" / "cqadupstack"


def flatten_cqad():
    subsets = list(map(
        lambda pth: pth.parts[-1],
        cqad_dir.glob("*")
    ))

    flatten_corpus = {}
    flatten_queries = {}
    flatten_qrels = {}

    for subset in subsets:
        subset_dir = cqad_dir / subset

        print(f"Processing cqadupstack/{subset}...")
        corpus, queries, qrels = GenericDataLoader(
            data_folder=str(subset_dir.absolute()),
        ).load("test")

        # prepend subset name to keys
        corpus = {
            f"{subset}_{k}": v
            for k, v in corpus.items()
        }
        queries = {
            f"{subset}_{k}": v
            for k, v in queries.items()
        }
        qrels = {
            f"{subset}_{k}": v
            for k, v in qrels.items()
        }

        if flatten_corpus.keys() & corpus.keys():
            raise ValueError("Duplicate keys found in corpus")

        flatten_corpus.update(corpus)
        flatten_queries.update(queries)
        flatten_qrels.update(qrels)


    print()


if __name__ == '__main__':
    flatten_cqad()