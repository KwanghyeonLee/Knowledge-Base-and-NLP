import json
from functools import reduce
from pathlib import Path
from typing import List, Dict, Tuple


if __name__ == '__main__':

    base = "msmarco"

    data_names = [
        "arguana",
        "climate-fever",
        # "cqadupstack",
        "dbpedia-entity",
        "fever",
        "fiqa",
        "hotpotqa",
        # "msmarco-v2",
        "nfcorpus",
        "nq",
        "quora",
        "scidocs",
        "scifact",
        "trec-covid",
        "webis-touche2020",
    ]

    total_rows = []
    for data_name in data_names:
        print(f"Processing {data_name}")
        dtype = "beir"

        root_dir = Path("/workspace")
        type_dir = root_dir / dtype
        target_data_dir = type_dir / data_name
        base_data_dir = type_dir / base

        base_idf_dir = base_data_dir / "idf"
        target_idf_dir = target_data_dir / "idf"

        base_dfs: List[int] = json.load(
            (base_idf_dir / "dfs.json").open("r")
        )
        base_toks: List[Tuple[int, int]] = list(
            sorted(
                map(
                    lambda i: (int(i[0]), base_dfs[int(i[0])]),
                    json.load(
                        (base_idf_dir / "id2token.json").open("r")
                    ).items()
                ),
                key=lambda x: x[1],
                reverse=True
            )
        )


        target_dfs: List[int] = json.load(
            (target_idf_dir / "dfs.json").open("r")
        )
        target_toks: List[Tuple[int, int]] = list(
            sorted(
                map(
                    lambda i: (int(i[0]), target_dfs[int(i[0])]),
                    json.load(
                        (target_idf_dir / "id2token.json").open("r")
                    ).items()
                ),
                key=lambda x: x[1],
                reverse=True
            )
        )

        default_mapped_values = dict(
            map(
                lambda i: (i, i),
                range(len(target_dfs)),
            )
        )
        # target -> base
        mapped_values = dict(
            map(
                lambda i: (i[0][0], i[1][0]),
                filter(
                    lambda i: i[0][1] > 0 and i[1][1] > 0,
                    zip(target_toks, base_toks)
                )
            ),
        )

        mapper = {
            # mapped values
            **default_mapped_values,
            # default
            **mapped_values,
        }

        with (target_idf_dir / "target2base.json").open("w") as fp:
            json.dump(mapper, fp)

        with (target_idf_dir / "base2target.json").open("w") as fp:
            json.dump(
                dict(
                    map(
                        lambda i: (i[1], i[0]),
                        mapper.items()
                    )
                ),
                fp
            )