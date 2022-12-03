import argparse
import pickle

import numpy as np

from datasets import disable_caching, load_from_disk
from faker import Faker
from functools import partial
from pathlib import Path
from pprint import pprint
from scrubadub import Scrubber
from scrubadub.detectors import CredentialDetector, TwitterDetector, UrlDetector
from scrubadub.filth import CreditCardFilth, EmailFilth, PhoneFilth, SocialSecurityNumberFilth
from squeakily.core import Pipeline
from squeakily.clean import replace_ip, normalize_whitespace, normalize_punctuation
from squeakily.filter import (
    check_char_repetition,
    check_flagged_words,
    check_stop_word_ratio,
    check_perplexity,
    check_word_number,
)
from squeakily.helpers import KenlmModel
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

# Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_dir",
    type=str,
    required=True,
    help="The directory where the data is stored.",
)
parser.add_argument(
    "--stats_path",
    type=str,
    required=True,
    help="The path to the stats file.",
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
    "Bible",
    "OtherWiki",
    "TheStack",
    "GithubDiffs"
]
dataset_cats = [x.name for x in data_dir.iterdir() if x.is_dir() and x.name not in ignored_datasets][:2]

scrubber = Scrubber()
scrubber.remove_detector(CredentialDetector)
scrubber.remove_detector(TwitterDetector)
scrubber.remove_detector(UrlDetector)

faker = Faker()
normalize_cleaning = lambda x: (
    scrubber.clean(x)
            .replace("{{EMAIL}}", EmailFilth.generate(faker))
            .replace("{{PHONE}}", PhoneFilth.generate(faker))
            .replace("{{SOCIAL_SECURITY_NUMBER}}", SocialSecurityNumberFilth.generate(faker))
            .replace("{{CREDIT_CARD}}", CreditCardFilth.generate(faker))
)

check_char_repetition_p = partial(check_char_repetition, char_repetition_threshold=0.6)
check_char_repetition_p.__name__ = "check_char_repetition"
check_stop_word_ratio_p = partial(check_stop_word_ratio, stop_word_threshold=0.8)
check_stop_word_ratio_p.__name__ = "check_stop_word_ratio"
check_flagged_words_p = partial(check_flagged_words, flagged_words_threshold=0.01)
check_flagged_words_p.__name__ = "check_flagged_words"

universal_filters = [
    check_char_repetition_p,
    check_flagged_words_p,
    check_stop_word_ratio_p,
]
universal_cleaners = [
    replace_ip,
    normalize_cleaning,
]
datasources = [
    {
        "dataset": load_from_disk(data_dir / k).select(range(10_000)),
        "name": k,
        "columns": ["text"],
        "filters": universal_filters,
        "cleaners": universal_cleaners,
    }
    for k in dataset_cats
]

stats_dict = pickle.load(open(args.stats_path, "rb"))
for ds in datasources:
    stats = stats_dict[ds["name"]]
    word_min, word_max = np.quantile(stats["word_count"]["lst"], [args.min_percentile, args.max_percentile])
    check_word_number_p = partial(check_word_number, min_word_threshold=word_min, max_word_threshold=word_max)
    check_word_number_p.__name__ = "check_word_number"
    ds["filters"].append(check_word_number_p)

    perplexity_max = np.quantile(stats["perplexity"]["lst"], args.max_percentile)
    check_perplexity_p = partial(check_perplexity, model=kenlm_model, perplexity_threshold=perplexity_max)
    check_perplexity_p.__name__ = "check_perplexity"
    ds["filters"].append(check_perplexity_p)

pprint(datasources)

pipeline = Pipeline(datasources)
pipeline.run()

pprint(pipeline.datasources)

# Save the resulting datasets
for name, ds in zip(dataset_cats, pipeline.datasources):
    ds["dataset"].save_to_disk(output_dir / name)