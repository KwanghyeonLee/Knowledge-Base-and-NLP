import gzip
import json
import os
import subprocess
from functools import partial
from pathlib import Path
from urllib.parse import unquote

import pandas as pd
import wget
from pyserini.search.lucene import LuceneSearcher
from tqdm import tqdm
import urllib.request


def show_progress(pbar, desc, block_num, block_size, total_size):
    if pbar is None:
        pbar = tqdm(total=total_size, desc=desc, unit="b", unit_scale=True)
    downloaded = block_num * block_size
    if downloaded < total_size:
        pbar.update(block_size)
    else:
        pbar.close()
        pbar = None
def download(url: str, output: Path):
    if not url.lower().startswith("http"):
        raise ValueError("url must start with http")
    file_name = unquote(url.split("/")[-1])
    try:
        pbar = None
        show_prog = partial(show_progress, pbar, f"Download {file_name}")
        urllib.request.urlretrieve(url, output, show_prog)  # nosec
    except Exception as e:
        raise RuntimeError(f"Download failed: {file_name}")


def fast_download(url, output_path):
    wget.download(url, out=output_path)

def download_with_aria2c(url, output_path):
    try:
        subprocess.run(['aria2c', '--quiet', url, '-o', output_path], check=True)
        print(f"Download complete: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Download failed: {e}")

def download_dataset(name : str):
    """_summary_
        Download the given dataset.
    Args:
        name (str): Name of the dataset.
    """
    # Set download path and file names
    corpus_name = f'beir-v1.0.0-{name}-flat-lucene8'
    query_name = f'topics.beir-v1.0.0-{name}.test.tsv.gz'
    qrel_name = f'qrels.beir-v1.0.0-{name}.test.txt'

    root_dir = Path("/workspace")
    tmp_dir = root_dir / "tmp"
    type_dir = root_dir / "beir"
    data_dir = type_dir / name
    data_dir.mkdir(exist_ok=True)

    # download doc
    # searcher = LuceneSearcher.from_prebuilt_index(corpus_name)
    # with (data_dir / "corpus.jsonl").open("w", encoding="utf-8") as fp:
    #     for i in tqdm(range(searcher.num_docs), desc=f'Download {name} (doc)'):
    #         doc = searcher.doc(i)
    #         data = json.loads(doc.raw())
    #         fp.write(json.dumps(data) + '\n')

    # download query
    (tmp_dir / name).mkdir(exist_ok=True)
    query_download_path = tmp_dir / name / 'query.test.tsv.gz'

    download(f'https://github.com/castorini/anserini/raw/cd6ac6fed364cd31606ebce65dd3b1bf83648e85/src/main/resources/topics-and-qrels/{query_name}',
              query_download_path)
    with gzip.open(query_download_path, 'rt') as fp:
        with (data_dir / "queries.jsonl").open("w", encoding='utf-8') as fp_out:
            for line in tqdm(fp, desc=f'Download {name} (query)'):
                qid, *text = line.strip().split('\t')
                text = ' '.join(text)
                data = {'_id': qid, 'text': text, 'metadata': {}}
                fp_out.write(json.dumps(data) + '\n')

    query_download_path.unlink()

    # download qrel
    qrels_path = data_dir / "qrels"
    qrels_path.mkdir(exist_ok=True)

    qrels_download_path = tmp_dir / name / 'qrel.test.txt'
    pbar = None
    download(f'https://github.com/castorini/anserini/raw/cd6ac6fed364cd31606ebce65dd3b1bf83648e85/src/main/resources/topics-and-qrels/{qrel_name}',
                  qrels_download_path)

    pd.read_csv(
        qrels_download_path,
        sep=' ',
        header=None,
        names=['query-id', "_", 'corpus-id', 'score'],
        usecols=['query-id', 'corpus-id', 'score']
    ).to_csv(qrels_path / "test.tsv", sep='\t', index=False)

    qrels_download_path.unlink()

if __name__ == '__main__':
    download_dataset("robust04")
    download_dataset("bioasq")

