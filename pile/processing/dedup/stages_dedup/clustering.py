"""
Given a list file each with minhashes as the entries generate from `generate_minhash.py`, perform clustering and deduplication.
The input files are generated from `generate_minhash.py` and should have the suffix `_minhash`.

Output is a series of files each is a boolean array with 'True' indicating the entry is a duplicate corresponding to the input files.
Output filename are the same as the input filename with the suffix `_filter_idx`.

Example:
Assume there are dataset file `StackExchange_minhash` and `AI4Code_minhash` in the list file `minhash_dataset_list.txt` (one path per line)
Outputs are `StackExchange_minhash_filter_idx` and `AI4Code_minhash_filter_idx` files in the same directory as the corresponding input files
`python clustering.py --minhash_dataset_list_file minhash_dataset_list.txt`
"""
from __future__ import annotations
from datasets import load_dataset, load_from_disk, Dataset, concatenate_datasets
import json
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author      : Chenghao Mou (mouchenghao@gmail.com)
# created     : 10/4/22

import gc
import hashlib
import logging
import multiprocessing as mp
import os
import random
import re
import struct
import time
import warnings
from collections import defaultdict
from itertools import tee
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import Tuple

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import datasets
    import numpy as np
    import typer
    from datasets import load_dataset
    from scipy.integrate import quad as integrate
    from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
datasets.logging.set_verbosity_error()
class UnionFind:
    def __init__(self):
        self.parent: Dict[int, int] = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        px = self.find(x)
        py = self.find(y)
        self.parent[px] = self.parent[py] = min(px, py)


if __name__ == "__main__":
    def run(
        minhash_dataset_list_file: str = typer.Option(..., help="minhash dataset paths to dedup. One path for each line"),  # noqa: E501
    ):
        global uf

        logging.basicConfig(level=logging.INFO)

        start_time = time.time()
        time_measures = {}

        # Checking the minhash_dataset_list_file is valid format
        minhash_dataset_paths = []
        with open(minhash_dataset_list_file, "r") as f:
            lines = f.read().splitlines()
            for line in lines:
                minhash_dataset_path = line.strip()
                minhash_dataset_paths.append(minhash_dataset_path)
                # check if the path is valid and is a directory
                assert os.path.isdir(minhash_dataset_path.strip())
                # check if the path is a minhash dataset (ends with _minhash)
                assert minhash_dataset_path.strip().endswith("_minhash")
                # check if it is a hugging face dataset format by checking if it has a dataset_info.json
                assert os.path.isfile(os.path.join(minhash_dataset_path.strip(), "dataset_info.json"))

        print('loading all the minhash datasets...')
        time_measures["load_minhash"] = time.time()
        minhash_datasets = {}
        offset_store = {}
        offset = 0
        for minhash_dataset_path in minhash_dataset_paths:
            offset_store[minhash_dataset_path] = offset
            dataset = load_from_disk(minhash_dataset_path.strip())
            minhash_datasets[minhash_dataset_path] = dataset
            # update the offset for the next dataset
            offset += len(dataset)

        # concatenate all the minhashes to one dataset
        embedded = concatenate_datasets(list(minhash_datasets.values()))
        time_measures["load_minhash"] = time.time() - time_measures["load_minhash"]

        print('output will be saved to the following files:')
        outputs = dict()
        for minhash_dataset_path in minhash_dataset_paths:
            
            print(minhash_dataset_path)
            print(minhash_dataset_path[:-len("_minhash")])
            outputs[minhash_dataset_path] = minhash_dataset_path[:-len("_minhash")] + "_filter_idx"
            print(outputs[minhash_dataset_path])

        time_measures["clustering"] = time.time()

        # get the first element of the dataset
        first = embedded[0]
        B = len(first['__signatures__'])
        batch_size: int = 10000
        for table_idx in range(B):
            new_hash_table = defaultdict(set)
            for i in tqdm(
                range(0, len(embedded), batch_size), dynamic_ncols=True, desc="Iterating MinHashes..."  # noqa: E501
            ):
                batch = embedded[i : i + batch_size]
                for tmp_idx, Hs in enumerate(batch["__signatures__"]):
                    #TODO check if it is correct
                    new_hash_table[Hs[table_idx]].add(i + tmp_idx)

            for cluster in new_hash_table.values():
                if len(cluster) <= 1:
                    continue
                idx = min(cluster)
                for x in cluster:
                    uf.union(x, idx)

        time_measures["clustering"] = time.time() - time_measures["clustering"]

        time_measures["filtering_idx"] = time.time()
        gc.freeze()
        gc.disable()
        final_data = embedded.map(
            function=lambda _, idx: {"__filter__": uf.find(idx) !=  idx},
            with_indices=True,
            num_proc=os.cpu_count(),
            new_fingerprint=str(random.getrandbits(128)),
            desc="Finding clusters...",
        )
        gc.enable()
        gc.collect()
        time_measures["filtering_idx"] = time.time() - time_measures["filtering_idx"]

        final_data = final_data.remove_columns(["__signatures__"])
        NUM_TRUE_LABEL = sum(final_data['__filter__'])
        print('Total Number of idx to be filtered', NUM_TRUE_LABEL)

        print('saving filter idx data...')
        time_measures["save"] = time.time()
        for minhash_dataset_path in minhash_dataset_paths:
            offset = offset_store[minhash_dataset_path]
            size = len(minhash_datasets[minhash_dataset_path])
            output = outputs[minhash_dataset_path]
            filter_idx = final_data.select(range(offset, offset + size))
            filter_idx.save_to_disk(output)
            print(f"saved to {output}")
        time_measures["save"] = time.time() - time_measures["save"]

        PAD = 32

        MINHASH_DATA_SIZE = len(embedded)

        for key, value in time_measures.items():
            logger.info(f"{key:<{PAD}}: {value:.2f} seconds")
        logger.info(
            f"{'Number of input minhash':<{PAD}}: {MINHASH_DATA_SIZE} "  # noqa: E501
        )
        logger.info(f"{'Duplicate Number':<{PAD}}: {NUM_TRUE_LABEL} ({NUM_TRUE_LABEL / MINHASH_DATA_SIZE:.2%})")  # noqa: E501
        logger.info(f"{'Total Time':<{PAD}}: {time.time() - start_time:.2f} seconds")
        logger.info(f"{'Filtered Index Dataset':<{PAD}}: {output}")
        logger.info(f"{'Finish generating filtered index from':<{PAD}}: {minhash_dataset_path}")
        logger.info("🤗 Happy Deduplicating 🤗")

    mp.set_start_method("fork", force=True)
    uf = UnionFind()
    typer.run(run)
