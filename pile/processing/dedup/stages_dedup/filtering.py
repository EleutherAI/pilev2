"""
Given a filter indicator file and a dataset file, filter the dataset.
The filter indicator file is a boolean array with 'True' indicating the entry is a duplicate corresponding to the input dataset.
The filter indicator file is generated from `clustering.py` and should have the suffix `_filter_idx`.
Example:
Assume there is a dataset file `StackExchange` and a filter indicator file `StackExchange_filter_idx`
Output will be `StackExchange_filtered`
`python filtering.py --dataset_path StackExchange --filter_idx_path StackExchange_filter_idx`
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

if __name__ == "__main__":
    def run(
        dataset_path: str = typer.Option(..., help="dataset to filter"),  # noqa: E501
        filter_idx_path: str = typer.Option(..., help="filter indicator dataset"),  # noqa: E501
        output_path: str = typer.Option(None, help="output path. If not set, would set to to dataset_path with suffix 'filtered'"),  # noqa: E501
    ):
        global uf

        logging.basicConfig(level=logging.INFO)
        # check if the filter_idx_path is ended with 'filter_idx'
        assert filter_idx_path.strip().endswith("filter_idx"), "filter_idx_path must end with 'filter_idx'"

        if output_path is None:
            output_path = dataset_path + "_filtered"
        typer.echo(f"output is saved to {output_path}")

        start_time = time.time()
        time_measures = {}
        # load the dataset from dataset_path
        time_measures["load_dataset"] = time.time()
        dataset = load_from_disk(dataset_path)
        time_measures["load_dataset"] = time.time() - time_measures["load_dataset"]
        INPUT_DATA_SIZE = len(dataset)
        # load the dataset from filter_idx_path
        time_measures["load_filter_idx"] = time.time()
        filter_idx_dataset = load_from_disk(filter_idx_path)
        time_measures["load_filter_idx"] = time.time() - time_measures["load_filter_idx"]
        # check if the dataset and filter_idx have the same number of rows
        assert len(dataset) == len(filter_idx_dataset), "dataset and filter_idx must have the same number of rows"

        # filtered out the dataset by filter_idx, if the filter_idx['__filter__'] is 1, then remove it, otherwise keep it
        dataset = dataset.add_column('__filter__', filter_idx_dataset["__filter__"])
        # we want to keep the data that has __filter__ == False
        dataset = dataset.filter(lambda x: x['__filter__'] == False)

        time_measures["save"] = time.time()
        dataset.save_to_disk(output_path)
        time_measures["save"] = time.time() - time_measures["save"]

        FINAL_DATA_SIZE = len(dataset)
        PAD = 32

        DUP_SIZE = INPUT_DATA_SIZE - FINAL_DATA_SIZE

        for key, value in time_measures.items():
            logger.info(f"{key:<{PAD}}: {value:.2f} seconds")
        logger.info(
            f"{'Number of input data':<{PAD}}: {INPUT_DATA_SIZE} "  # noqa: E501
        )
        logger.info(
            f"{'Data Number (after)':<{PAD}}: {FINAL_DATA_SIZE} ({FINAL_DATA_SIZE / INPUT_DATA_SIZE:.2%})"  # noqa: E501
        )
        logger.info(f"{'Duplicate Number':<{PAD}}: {DUP_SIZE} ({DUP_SIZE / INPUT_DATA_SIZE:.2%})")  # noqa: E501
        logger.info(f"{'Total Time':<{PAD}}: {time.time() - start_time:.2f} seconds")
        logger.info(f"{'Finish generating filtered dataset from':<{PAD}}: {dataset_path}")
        logger.info(f"{'Finish saving filtered dataset to':<{PAD}}: {output_path}")
        logger.info("🤗 Happy Deduplicating 🤗")

    mp.set_start_method("fork", force=True)
    typer.run(run)
