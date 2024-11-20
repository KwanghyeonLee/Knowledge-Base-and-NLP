import json
import math
from pathlib import Path

if __name__ == '__main__':
    datasets = [
        'scidocs',
        'scifact',
        'fiqa',
        'nfcorpus',
        'robust04',
        "trec-covid",
        "arguana",
        "webis-touche2020"
    ]

    for dataset in datasets:
        idf_dir = Path(f'/workspace/beir/{dataset}/idf')
        idf_file = idf_dir / 'idfs.json'
        idfs = json.loads(idf_file.read_text())

        # measure entropy of idfs
        idf_entropy = sum([-idf * math.log2(idf) for idf in idfs])
        print(f'{dataset:20s} idf entropy: {idf_entropy:10.2f}')

    for dataset in datasets:
        tf_dir = Path(f'/workspace/beir/{dataset}/tf')
        tf_file = tf_dir / 'tfs.json'
        tfs = json.loads(tf_file.read_text())
        tfs = [tf + 1 for tf in tfs]
        total_count = sum(tfs)
        tfs = [tf / total_count for tf in tfs]
        # measure entropy of tfs
        tf_entropy = sum([-tf * math.log2(tf) for tf in tfs])
        print(f'{dataset:20s} tf  entropy: {tf_entropy:10.2f}')

    for dataset in datasets:
        tf_dir = Path(f'/workspace/beir/{dataset}/tf')
        tf_file = tf_dir / 'tfs.json'
        tfs = json.loads(tf_file.read_text())
        tfs = [tf + 1 for tf in tfs]
        total_count = sum(tfs)
        expected_count = total_count / len(tfs)

        chi2 = sum([(tf - expected_count) ** 2 for tf in tfs])
        print(f'{dataset:20s} chi2: {chi2:15.2f}')


