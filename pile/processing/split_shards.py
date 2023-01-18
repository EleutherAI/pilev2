import datasets
import logging
import pathlib
import argparse
import time
from tqdm import tqdm

logging.basicConfig(
    level = logging.INFO,
    format=  '%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename=f"logs/stats_{time.time()}_split_group.log")

logger = logging.getLogger(__name__)


def log_both(msg):
    logger.info(msg)
    print(msg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--shard_num",type=int,default=10)
    args = parser.parse_args()
    log_both("Starting to split shards")
    dataset = datasets.load_from_disk(args.input_dir)
    log_both(f"Sucessfully loaded the dataset from {args.input_dir}.")

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for num_shard in tqdm(range(args.shard_num)):
        shard_dataset = dataset.shard(num_shards=args.shard_num,index=num_shard)
        shard_dataset.save_to_disk(output_dir/f"shard_{num_shard}")
        log_both(f"Saved shard {num_shard} to {output_dir/f'shard_{num_shard}'}")
    
    log_both("Finished splitting shards")