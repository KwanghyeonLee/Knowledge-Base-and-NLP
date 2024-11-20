import json
import logging
from collections import defaultdict
from pathlib import Path

import pandas as pd
import typer
from orjson import orjson
from tqdm.rich import tqdm

import warnings
warnings.filterwarnings('ignore')

workspace_dir = Path("/workspace")
amnesia_dir = workspace_dir / "amnesia"

def load_idf(experiment_name, data_name):
    print("Loading IDF...")
    with (amnesia_dir / experiment_name / data_name / f"idfs.json").open() as f:
        idf = orjson.loads(f.readline())
    return idf



def extract_topk(idf_score, mlm_score, k=100):
    idf_score = sorted(enumerate(idf_score), key=lambda x: x[1], reverse=True)[:k]
    mlm_score = sorted(enumerate(mlm_score), key=lambda x: x[1], reverse=True)[:k]

    idf_token_ids = set([x[0] for x in idf_score])
    mlm_token_ids = set([x[0] for x in mlm_score])

    return idf_token_ids, mlm_token_ids

def measure_intersection(seta, setb):
    assert len(seta) == len(setb)
    return len(seta.intersection(setb)) / len(seta)

def measure_jaccard_distance(seta, setb):
    return len(seta.intersection(setb)) / len(seta.union(setb))

def measure_missed(seta, setb):
    assert len(seta) == len(setb)
    return len(seta.difference(setb)) / len(seta)

def get_mean(lst):
    return sum(lst) / len(lst)

def get_min(lst):
    return min(lst)

def get_max(lst):
    return max(lst)

def main_run(k: int = 100):
    expr_dir = amnesia_dir / f"k={k}"
    expr_dir.mkdir(exist_ok=True, parents=True)
    log_writer = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(expr_dir / f"measure_amnesia_fiqa.log"),
            logging.StreamHandler()
        ]
    )
    log_writer.info("==========================================")
    log_writer.info(f"Start measuring Amnesia with k={k} ...")
    log_writer.info("==========================================")

    experiments = [
        "ours-gpl",
        "ours-ramda",
        "gpl-base",
        "gpl-ramda",
    ]

    datasets = [
        # "nfcorpus",
        # "scifact",
        # "scidocs"
        "fiqa"
    ]

    overall_jaccard = defaultdict(dict)
    overall_intersection = defaultdict(dict)
    overall_missed = defaultdict(dict)

    overall_min_jaccard = defaultdict(dict)
    overall_min_intersection = defaultdict(dict)
    overall_min_missed = defaultdict(dict)

    overall_max_jaccard = defaultdict(dict)
    overall_max_intersection = defaultdict(dict)
    overall_max_missed = defaultdict(dict)

    for dataset in datasets:
        for experiment in experiments:
            log_writer.info(f"Dataset: {dataset}, Experiment: {experiment}")

            # intersections = {}
            jaccards = {}
            # misseds = {}

            idf = load_idf(experiment, dataset)
            print("Measuring line count of MLM...")
            with (amnesia_dir / experiment / dataset / f"mlm.jsonl").open("rbU") as f:
                num_lines = sum(1 for _ in f)

            with (amnesia_dir / experiment / dataset / f"mlm.jsonl").open() as f:
                for line in tqdm(f, total=num_lines, desc="Measuring Amnesia"):
                    obj = orjson.loads(line)
                    idf_toks, mlm_toks = extract_topk(idf, obj["mlm"], k=k)
                    jaccard = measure_jaccard_distance(idf_toks, mlm_toks)
                    jaccards[obj["id"]] = jaccard


            # overall_intersection[dataset][experiment] = get_mean(list(intersections.values()))
            overall_jaccard[dataset][experiment] = get_mean(list(jaccards.values()))
            # overall_missed[dataset][experiment] = get_mean(list(misseds.values()))

            # overall_min_intersection[dataset][experiment] = get_min(list(intersections.values()))
            # overall_min_jaccard[dataset][experiment] = get_min(list(jaccards.values()))
            # overall_min_missed[dataset][experiment] = get_min(list(misseds.values()))
            #
            # overall_max_intersection[dataset][experiment] = get_max(list(intersections.values()))
            # overall_max_jaccard[dataset][experiment] = get_max(list(jaccards.values()))
            # overall_max_missed[dataset][experiment] = get_max(list(misseds.values()))

            # log_writer.info(f"Intersection: {get_mean(list(intersections.values()))}")
            log_writer.info(f"Jaccard: {get_mean(list(jaccards.values()))}")
            # log_writer.info(f"Missed: {get_mean(list(misseds.values()))}")
            log_writer.info("==========================================")

    # pd.DataFrame(overall_intersection).to_csv(expr_dir / "overall_intersection_fiqa.csv")
    pd.DataFrame(overall_jaccard).to_csv(expr_dir / "overall_jaccard_fiqa.csv")
    # pd.DataFrame(overall_missed).to_csv(expr_dir / "overall_missed_fiqa.csv")

    # pd.DataFrame(overall_min_intersection).to_csv(expr_dir / "overall_min_intersection_fiqa.csv")
    # pd.DataFrame(overall_min_jaccard).to_csv(expr_dir / "overall_min_jaccard_fiqa.csv")
    # pd.DataFrame(overall_min_missed).to_csv(expr_dir / "overall_min_missed_fiqa.csv")
    #
    # pd.DataFrame(overall_max_intersection).to_csv(expr_dir / "overall_max_intersection_fiqa.csv")
    # pd.DataFrame(overall_max_jaccard).to_csv(expr_dir / "overall_max_jaccard_fiqa.csv")
    # pd.DataFrame(overall_max_missed).to_csv(expr_dir / "overall_max_missed_fiqa.csv")

    import requests
    requests.post(
        'https://hooks.slack.com/services/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',
        json={'text': f'k={k} amnesia Done'}
    )

if __name__ == '__main__':
    main_run(1000)
    # typer.run(main_run)