import argparse
import gc
import pickle

import numpy as np

from datasets import load_from_disk
from functools import partial
from pathlib import Path
from pprint import pprint
from squeakily.core import Pipeline
from squeakily.filter import check_compression_ratio

# Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_dir",
    type=str,
    required=True,
    help="The directory where the data is stored.",
)
parser.add_argument(
    "--min_percentile",
    type=float,
    default=0.01,
    help="The minimum percentile to use for the threshold.",
)
parser.add_argument(
    "--max_percentile",
    type=float,
    default=0.99,
    help="The maximum percentile to use for the threshold.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    required=True,
    help="The directory where the output should be stored.",
)
args = parser.parse_args()

# create the output directory if it does not exist
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

# Get all the files in the data directory
data_dir = Path(args.data_dir)
ignored_datasets = [
    "AMPS",
    "DMMath",
    "Enwiki",
    "EuroParliamentProceedings",
    "Gutenberg",
    "OtherWiki",
    "TheStack",
    "GithubDiffs"
]
dataset_cats = ["DevDocs", "Opensubtitles", "GithubIssues_ver2"]
# dataset_cats = [x.name for x in data_dir.iterdir() if x.is_dir() and x.name not in ignored_datasets][:2]

datasources = (
    {
        "dataset": load_from_disk(data_dir / k, keep_in_memory=False),
        "name": k,
        "columns": ["text"],
        "filters": [check_compression_ratio],
        "cleaners": [],
    }
    for k in dataset_cats
)
for idx, ds in enumerate(datasources):
    name = ds["name"]
    pipeline = Pipeline([ds])
    pipeline.run(dry_run=True, num_proc=96)
    new_ds = pipeline.datasources[0]["dataset"]
    start_size = len(new_ds)
    compression_ratios = new_ds["check_compression_ratio_criteria"]
    min_compression_ratio = np.quantile(compression_ratios, args.min_percentile)
    new_ds = new_ds.filter(
        lambda x: x["check_compression_ratio_criteria"] > min_compression_ratio,
        batched=True,
        num_proc=32,
    )
    end_size = len(new_ds)
    print(f"Dataset {name} went from {start_size} to {end_size} rows.")
    new_ds.save_to_disk(output_dir / name)
    del ds, new_ds, pipeline
    gc.collect()
