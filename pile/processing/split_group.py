import datasets
import multiprocessing as mp
import os
import pathlib
import json
import argparse
import logging
import time
import ast

logging.basicConfig(
    level = logging.INFO,
    format=  '%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename=f"logs/stats_{time.time()}_split_group.log")

logger = logging.getLogger(__name__)
meta_match = {
    "arXiv_ver2" :  lambda example : ast.literal_eval(example["meta"]["source"]) == "arXiv_out/",
    "PubMed_ver2" : lambda example : ast.literal_eval(example["meta"]["source"]) == "PubMedDataset",
    "Gutenberg_ver2" : lambda example : ast.literal_eval(example["meta"]["source"]) == "Project Gutenberg",
    "FreeLaw_Options_ver2" : lambda example : "date_created" in example.keys(),
    "UbuntuIRC_ver2" : lambda example : ast.literal_eval(example["meta"]["source"]) == "Ubuntu IRC",
    "Enwiki_ver2" : lambda example : "wikidata_id" in example.keys(),
    "EuroParliamentProceedings_ver2" : lambda example : "language" in example.keys(),
    "USPTO_ver2" : lambda example : ast.literal_eval(example["meta"]["source_data"]) == "USPTO-Application",
    "PileOfLaw_ver2" : lambda example : "dataset" in example.keys(),
    "OtherWiki_ver2" : lambda example : "wiki_source" in example.keys(),
    "S2ORC_ver2" : lambda example : ast.literal_eval(example["meta"]["source"]) == "S2ORC",
}



def filter_subset(example,subset_key:str):
    if meta_match[subset_key] in example["meta"]:
        return True
    else:
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--stats_output_dir", type=str)
    parser.add_argument("--num_workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    dataset = datasets.load_from_disk(args.input_dir)
    stats_dict = {}
    for subset_key in meta_match.keys():
        logger.info(f"Starting to filter {subset_key}")
        print(f"Starting to filter {subset_key}")
        subset_dataset = dataset.filter(meta_match[subset_key], num_proc=args.num_workers)
        length = len(subset_dataset)
        stats_dict[subset_key] = length
        logger.info(f"Length of {subset_key} is {length}")
        print(f"Length of {subset_key} is {length}")
        output_dir = pathlib.Path(args.output_dir)/subset_key
        output_dir.mkdir(parents=True, exist_ok=True)
        subset_dataset.save_to_disk(output_dir)
        logger.info(f"Saved {subset_key} to {output_dir}")
        print(f"Saved {subset_key} to {output_dir}")


    stats_output_dir = pathlib.Path(args.stats_output_dir)
    stats_output_dir.mkdir(parents=True, exist_ok=True)
    with open(stats_output_dir/"stats.json","w") as f:
        json.dump(stats_dict,f,indent=2)
    