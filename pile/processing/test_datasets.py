import argparse
import random

from datasets import load_from_disk
from pathlib import Path
from pprint import pprint
from tqdm.auto import tqdm

# set the seed
random.seed(115)

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
dataset_cats = [x.name for x in data_dir.iterdir() if x.is_dir()]

for k in tqdm(dataset_cats):
    try:
        ds = load_from_disk(data_dir / k)
        # select a random subset of the dataset
        print(k)
        print(len(ds))
        ds = ds.select(random.sample(range(len(ds)), 100))

        # write the subset to a text file
        with open(output_dir / f"{k}.txt", "w") as f:
            for i in tqdm(range(len(ds))):
                f.write(ds[i]["text"] + "\n\n")
    except Exception as e:
        print(e)
        continue