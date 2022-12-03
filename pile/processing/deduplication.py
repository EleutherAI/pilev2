import argparse

from datasets import disable_caching, load_from_disk
from functools import partial
from pathlib import Path
from pprint import pprint
from squeakily.core import Pipeline
from squeakily.filter import minhash_dedup
disable_caching()

# Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_dir",
    type=str,
    required=True,
    help="The directory where the data is stored.",
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
dataset_cats = [x.name for x in data_dir.iterdir() if x.is_dir()][:2]

datasources = [
    {
        "dataset": load_from_disk(data_dir / k).select(range(10_000)),
        "name": k,
        "columns": ["text"],
        "filters": [],
        "cleaners": [],
    }
    for k in dataset_cats
]

pprint(datasources)

pipeline = Pipeline(datasources)
pipeline.run()

pprint(pipeline.datasources)

# Save the resulting datasets
for name, ds in zip(dataset_cats, pipeline.datasources):
    ds["dataset"].save_to_disk(output_dir / name)

# # Determine the number of shards per datasource to split the data into for saving
num_files_per_shard = 10_000
for name, ds in zip(dataset_cats, pipeline.datasources):
    ds_len = len(ds["dataset"])
    num_shards = ds_len // num_files_per_shard
    if num_shards == 0:
        num_shards = 1
    ds_shards = [ds["dataset"].shard(num_shards, i, contiguous=True) for i in range(num_shards)]
    for i, shard in enumerate(ds_shards):
        path = output_dir / f"{name}_shard_{i}.jsonl.zst"
        shard.to_json(
            path,
            lines=True,
            orient="records",
        )