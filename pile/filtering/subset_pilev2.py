from datasets import load_from_disk
import multiprocessing as mp
import os
import logging
from tqdm import tqdm
import argparse
import time
import pathlib
import random
import json

logging.basicConfig(
    level = logging.INFO,
    format=  '%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename=f"pile/logs/the_pile_subset_{time.time()}.log")
logger = logging.getLogger(__name__)

def subset_dataset_fn(dataset,percentage:float=10.0):
    if len(dataset) > 10:
        percentage_total = int(len(dataset)*percentage/100)
        idxs = random.sample(range(len(dataset)),percentage_total)
        
        total_len = len(dataset)
        dataset = dataset.select(idxs)
        #idxs to dataset column
        print(f"Slicing 10% which is {percentage_total} |  {len(dataset)} out of {total_len}")
        logger.info(f"Slicing 10% which is {percentage_total} | {len(dataset)} out of {total_len}")
        #dataset.add_column("subset_idx",idxs) This takes painfully too long So making a quick replacement.
        print(dataset)
        logger.info(dataset)
        return dataset,idxs
    else:
        return dataset,[]


def write_to_json(path,list_of_inds:list):
    with open(os.path.join(path,"subset_indices.json"),"w") as f:
        json.dump({"idx" : list_of_inds},f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--percentage",type=float,default=10.0)
    parser.add_argument("--num_workers", type=int, default=128)
    args = parser.parse_args()

    input_sub_dirs = os.listdir(args.input_dir)

    logger.info("Loading dataset")
    for sub_dir in tqdm(input_sub_dirs,leave=False):
        try:
            if sub_dir != "non_local_dedup":
                path_sub_dir = os.path.join(args.input_dir, sub_dir)
                print(path_sub_dir)
                logger.info(path_sub_dir)
                dataset = load_from_disk(path_sub_dir)
                logger.info(f"Loaded dataset from {path_sub_dir}")
                print(f"Loaded dataset from {path_sub_dir}")
                subset_dataset,subset_idx = subset_dataset_fn(dataset,percentage=10.0)
                logger.info(f"Subsetting dataset to {args.percentage}%")

                logger.info("Saving dataset")
                output_dir = pathlib.Path(os.path.join(args.output_dir, sub_dir))
                output_dir.mkdir(parents=True, exist_ok=True)
                subset_dataset.save_to_disk(args.output_dir)
                write_to_json(output_dir,subset_idx)
                logger.info(f"Saved dataset at {args.output_dir}")
            else:
                logger.info(f"The subdir is non_local_dedup")
                print(f"The subdir is non_local_dedup")
                non_local_sub_dirs = os.listdir(os.path.join(args.input_dir,sub_dir))
                for non_local_sub_dir in tqdm(non_local_sub_dirs,leave=False):
                    path_non_local_sub_dir = os.path.join(args.input_dir,sub_dir, non_local_sub_dir)
                    print(path_sub_dir)
                    logger.info(path_sub_dir)
                    dataset = load_from_disk(path_non_local_sub_dir)
                    logger.info(f"Loaded dataset from {path_non_local_sub_dir}")
                    print(f"Loaded dataset from {path_non_local_sub_dir}")
                    subset_dataset = subset_dataset(dataset,percentage=10.0)
                    logger.info(f"Subsetting dataset to {args.percentage}%")

                    logger.info("Saving dataset")
                    output_dir = pathlib.Path(os.path.join(args.output_dir,sub_dir ,non_local_sub_dir))
                    output_dir.mkdir(parents=True, exist_ok=True)
                    subset_dataset.save_to_disk(args.output_dir)
                    write_to_json(output_dir,subset_idx)
                    logger.info(f"Saved dataset at {args.output_dir}")
        except PermissionError:
            logger.info(f"Permission Error at {sub_dir}")
            print(f"Permission Error at {sub_dir}")
        except FileNotFoundError:
            logger.info(f"FileNotFoundError  at {sub_dir}")
            print(f"FileNotFoundError  at {sub_dir}")
