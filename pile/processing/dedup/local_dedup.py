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

for k in dataset_cats:
    datasources = [
        {
            "dataset": load_from_disk(data_dir / k),
            "name": k,
            "columns": ["text"],
            "filters": [],
            "cleaners": [],
        }
    ]

    pprint(datasources)

    pipeline = Pipeline(datasources)
    pipeline.run(global_filters=[minhash_dedup])

    pprint(pipeline.datasources)

    # Save the resulting datasets
    pipeline.datasources[0]["dataset"].save_to_disk(output_dir / k)