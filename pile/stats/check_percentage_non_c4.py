import datasets
import multiprocessing as mp
import os
import pathlib
import json
import argparse
import logging
import time

logging.basicConfig(
    level = logging.INFO,
    format=  '%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename=f"logs/stats_{time.time()}.log")


meta_match = {
    "arxiv" : 
    [ "arxiv_out" ],
    "pubmed" : 
    ["PubMedDataset"]
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
    parser.add_argument("--num_workers", type=int, default=128)
    args = parser.parse_args()

    dataset = datasets.load_from_disk(args.input_dir)
    stats_dict = {}
    for subset_key in meta_match.keys():
        subset_dataset = dataset.filter(filter_subset,fn_kwargs={"subset_key":subset_key} num_proc=args.num_workers)
        length = len(subset_dataset)
        stats_dict[subset_key] = length
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir/"stats.json","w") as f:
        json.dump(stats_dict,f,indent=2)
    