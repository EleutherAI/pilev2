from datasets import load_from_disk,load_dataset
import multiprocessing as mp
import os
import logging
from tqdm import tqdm
from transformers import GPTNeoXTokenizerFast
import argparse
import time
import pathlib
import ast
import random

logging.basicConfig(
    level = logging.INFO,
    format=  '%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename=f"logs/the_pile_subset_{time.time()}.log")
logger = logging.getLogger(__name__)

def subset_dataset(dataset,percentage:float=10.0):
    
    percentage_total = int(len(dataset)*percentage/100)
    idxs = random.sample(range(len(dataset)),percentage_total)
    dataset = dataset.select(idxs)
    #idxs to dataset column
    dataset["subset_idx"] = idxs
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--percentage",type=float,default=10.0)
    parser.add_argument("--num_workers", type=int, default=128)
    args = parser.parse_args()

    input_sub_dirs = os.listdir(args.input_dir)

    logger.info("Loading dataset")
    for sub_dir in tqdm(input_sub_dirs):
        path_sub_dir = os.path.join(args.input_dir, sub_dir)
        dataset = load_from_disk(path_sub_dir)
        logger.info(f"Loaded dataset from {path_sub_dir}")
        subset_dataset = subset_dataset(dataset,percentage=10.0)
        logger.info(f"Subsetting dataset to {args.percentage}%")

        logger.info("Saving dataset")
        output_dir = pathlib.Path(os.path.join(args.output_dir, sub_dir))
        output_dir.mkdir(parents=True, exist_ok=True)
        subset_dataset.save_to_disk(args.output_dir)
        logger.info(f"Saved dataset at {args.output_dir}")