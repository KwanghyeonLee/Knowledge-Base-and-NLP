import json
from functools import reduce

from pathlib import Path
import gensim
import numpy as np
import pandas as pd
import torch
import transformers
import typer
from einops import repeat, rearrange
from numpy import inf
from torch import log, relu, tensor, nn
from tqdm.rich import tqdm

def main(
        data_name: str,
        cuda_id: int = 0,
        dtype: str = "gpl"
):
    print(f"Processing {data_name}")

    root_dir = Path("/workspace")
    target_dir = root_dir / dtype

    gpl_training = pd.read_csv(target_dir / data_name / "gpl-training-data.tsv", sep="\t", names=["qid", "pid", "nid", "label"])

    total = tensor([], device=f"cuda:{cuda_id}", dtype=torch.float)
    files = list((target_dir / data_name / "curriculum").iterdir())
    for pt in tqdm(files, desc="Loading curriculum"):
        total = torch.cat(
            [
                total,
                torch.load(pt).to(f"cuda:{cuda_id}")
            ],
            0
        )

    sorted_indices = total[:, 0].sort()[1]
    total = total[sorted_indices]

    curriculum_training = gpl_training
    curriculum_training["difficulty"] = total[:, 1].cpu().numpy()
    curriculum_training["abs_label"] = abs(curriculum_training["label"])
    curriculum_training.sort_values(by=["abs_label", "difficulty"], ascending=[False, True], inplace=True)
    curriculum_training.sort_values(by=["difficulty"], inplace=True)
    curriculum_training = curriculum_training[["qid", "pid", "nid", "label"]]

    curriculum_training.to_csv(
        target_dir / data_name / "curriculum-gpl-training-data.tsv", sep="\t", index=False, header=False
    )


if __name__ == '__main__':
    typer.run(main)
    # main("nfcorpus", 0, "gpl")


