from datasets import load_from_disk,save_to_disk,load_dataset
import multiprocessing as mp
import os
import logging
from tqdm import tqdm
from transformers import GPTNeoXTokenizerFast
import argparse

logging.basicConfig(
    level = logging.INFO,
    format=  '%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename="/fsx/home-reshinth/work/the_stack/processing.log")
logger = logging.getLogger(__name__)


FLAG_FLAT_LIST = ["HTML","JSON","JSON5","JSONLD","YAML","XML","HTTP","CSV","SVG"]

def filter_flat_langs(example):
    if example["lang"] not in FLAG_FLAT_LIST:
        return True
    else:
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--num_workers", type=int, default=128)
    args = parser.parse_args()

    logger.info("Loading dataset")
    dataset = load_from_disk("the_stack", data_dir=args.input_dir)
    logger.info("Loaded dataset")

    logger.info("Filtering flat langs")
    dataset = dataset.filter(filter_flat_langs, num_proc=args.num_workers)
    logger.info("Filtered flat langs")

    logger.info("Saving dataset")
    dataset.save_to_disk(args.output_dir)
    logger.info(f"Saved dataset at {args.output_dir}")
