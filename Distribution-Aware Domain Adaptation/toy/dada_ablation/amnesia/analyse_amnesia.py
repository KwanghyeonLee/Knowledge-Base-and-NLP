from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

workspace_dir = Path("/workspace")
amnesia_dir = workspace_dir / "amnesia"


experiment_overleaf = {
    "gpl-base": "GPL",
    "ours-gpl": "GPL + DADA",
    "gpl-ramda": "GPL$^+$",
    "ours-ramda": "GPL$^+$ + DADA",
}


def convert_2d_table_to_1d_table(df: pd.DataFrame, dtype: str = "avg"):
    drop_exprs = ["scidocs"]
    if "fiqa" in df.columns:
        drop_exprs.append("fiqa")
    df = df.drop(drop_exprs, axis=1)
    df = df.reindex([2,0,3,1])
    experiments = df["Unnamed: 0"].tolist()
    new_experiments = [experiment_overleaf[expr] for expr in experiments]

    df = df.set_index("Unnamed: 0")
    datasets = df.columns.tolist()

    new_df = pd.DataFrame(columns=["experiment", "dataset", dtype])
    for experiment, new_experiment in zip(experiments, new_experiments):
        for dataset in datasets:
            new_df = new_df.append({"experiment": new_experiment, "dataset": dataset, dtype: df[dataset][experiment]}, ignore_index=True)


    return new_df


def errplot(x, y, yerr, hue, **kwargs):
    data = kwargs.pop('data')
    p = data.pivot_table(index=x, columns=hue, values=y, aggfunc='mean')
    err = data.pivot_table(index=x, columns=hue, values=yerr, aggfunc='mean')
    p.plot(kind='bar', yerr=err, ax=plt.gca(), **kwargs)


def main_run(k: int = 100):
    expr_dir = amnesia_dir / f"k={k}"

    print("==========================================")
    print(f"Start measuring Amnesia with k={k} ...")
    print("==========================================")

    overall_avg_intersection_df = convert_2d_table_to_1d_table(pd.read_csv(expr_dir / "overall_intersection.csv"))
    overall_min_intersection_df = convert_2d_table_to_1d_table(pd.read_csv(expr_dir / "overall_min_intersection.csv"), "min")
    overall_max_intersection_df = convert_2d_table_to_1d_table(pd.read_csv(expr_dir / "overall_max_intersection.csv"), "max")
    overall_intersection_df = pd.merge(
        pd.merge(overall_avg_intersection_df, overall_min_intersection_df, on=["experiment", "dataset"]),
        overall_max_intersection_df, on=["experiment", "dataset"]
    )

    overall_avg_jaccard_df = convert_2d_table_to_1d_table(pd.read_csv(expr_dir / "overall_jaccard.csv"))
    overall_jaccard_df = overall_avg_jaccard_df
    # overall_min_jaccard_df = convert_2d_table_to_1d_table(pd.read_csv(expr_dir / "overall_min_jaccard.csv"), "min")
    # overall_max_jaccard_df = convert_2d_table_to_1d_table(pd.read_csv(expr_dir / "overall_max_jaccard.csv"), "max")
    # overall_jaccard_df = pd.merge(
    #     pd.merge(overall_avg_jaccard_df, overall_min_jaccard_df, on=["experiment", "dataset"]),
    #     overall_max_jaccard_df, on=["experiment", "dataset"]
    # )

    overall_avg_missed_df = convert_2d_table_to_1d_table(pd.read_csv(expr_dir / "overall_missed.csv"))
    overall_min_missed_df = convert_2d_table_to_1d_table(pd.read_csv(expr_dir / "overall_min_missed.csv"), "min")
    overall_max_missed_df = convert_2d_table_to_1d_table(pd.read_csv(expr_dir / "overall_max_missed.csv"), "max")
    overall_missed_df = pd.merge(
        pd.merge(overall_avg_missed_df, overall_min_missed_df, on=["experiment", "dataset"]),
        overall_max_missed_df, on=["experiment", "dataset"]
    )

    import seaborn as sns
    sns.set_theme(style="whitegrid")

    # Draw a nested barplot by species and sex
    g = sns.catplot(
        data=overall_jaccard_df,
        kind="bar",
        x="dataset", y="avg", hue="experiment",
        alpha=1,
        height=6,
    )

    g.despine(left=True)
    g.set_axis_labels("Datasets", "Jaccard Similarity between IDF and Model's MLM output")
    # g.fig.suptitle(f"Amnesia with k={k}")
    g.legend.set_title("Experiment")


    sns.move_legend(g, "center right")

    plt.show()
    print()




    print()

if __name__ == '__main__':
    # main_run(500)
    main_run(1000)
    # main_run(2000)
    # main_run(5000)
    # typer.run(main_run)