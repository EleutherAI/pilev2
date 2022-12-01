import argparse

from datasets import disable_caching, load_from_disk
from functools import partial
from pathlib import Path
from pprint import pprint
from squeakily.core import Pipeline
from squeakily.filter import (
    check_char_repetition,
    check_flagged_words,
    check_perplexity,
    check_language,
    check_word_number,
    minhash_dedup
)
from squeakily.helpers import FastTextLanguageDetector, KenlmModel
disable_caching()

kenlm_model = KenlmModel.from_pretrained(
    model_dataset="wikipedia",
    language="en",
    lower_case=True,
    remove_accents=True,
    normalize_numbers=True,
    punctuation=1,
)
check_perplexity_p = partial(check_perplexity, model=kenlm_model)
check_perplexity_p.__name__ = "check_perplexity"

fasttext_model = FastTextLanguageDetector.from_pretrained()
check_language_p = partial(check_language, model=fasttext_model)
check_language_p.__name__ = "check_language"

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

print(dataset_cats)
universal_filters = [
    check_char_repetition,
    check_flagged_words,
    check_perplexity_p,
    check_language_p,
    check_word_number,
]
datasources = [
    {
        "dataset": load_from_disk(data_dir / k).select(range(10_000)),
        "columns": ["text"],
        "filters": universal_filters,
        "cleaners": [],
    }
    for k in dataset_cats
]

pprint(datasources)

pipeline = Pipeline(datasources)
pipeline.run(global_filters=[minhash_dedup])

# Determine the number of shards per datasource to split the data into for saving
num_files_per_shard = 10_000
for name, ds in zip(dataset_cats, pipeline.datasources):
    ds_len = len(ds["dataset"])
    num_shards = ds_len // num_files_per_shard
    if shards == 0:
        shards = 1
    ds_shards = [ds["dataset"].shard(num_shards, i, contiguous=True) for i in range(num_shards)]
    for i, shard in enumerate(ds_shards):
        path = output_dir / f"{name}_shard_{i}.jsonl.zst"
        shard.to_json(
            path,
            lines=True,
            orient="records",
            compression="zstd",
        )