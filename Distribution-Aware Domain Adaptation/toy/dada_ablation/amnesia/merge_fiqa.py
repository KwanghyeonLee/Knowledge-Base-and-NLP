from pathlib import Path


import pandas as pd

import warnings
warnings.filterwarnings('ignore')

workspace_dir = Path("/workspace")
amnesia_dir = workspace_dir / "amnesia"

def main_run(k: int = 100):
    expr_dir = amnesia_dir / f"k={k}"
    overall_df = pd.read_csv(expr_dir / "overall_jaccard.csv")
    # overall_df.drop(["Unnamed: 0.1", "fiqa", "fiqa_x", "fiqa_y"], axis=1, inplace=True)
    fiqa_df = pd.read_csv(expr_dir / "overall_jaccard_fiqa.csv")

    overall_df = overall_df.merge(
        fiqa_df, on=["Unnamed: 0"],
    )

    overall_df.to_csv(expr_dir / "overall_jaccard.csv", index=False)

if __name__ == '__main__':
    main_run(k=1000)