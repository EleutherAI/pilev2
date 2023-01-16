#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author      : Chenghao Mou (mouchenghao@gmail.com)
# created     : 10/4/22
from __future__ import annotations

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
from jax import random
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import Tuple

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import datasets
    # import numpy as np
    import jax.numpy as np
    import typer
    from datasets import load_dataset
    from scipy.integrate import quad as integrate
    from tqdm import tqdm


SEED = 42
NON_ALPHA = re.compile("[^A-Za-z_0-9]")
RNG = random.PRNGKey(0)

RNG = np.random.RandomState(SEED)
MAX_HASH = np.uint64((1 << 32) - 1)
MERSENNE_PRIME = np.uint64((1 << 61) - 1)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
datasets.logging.set_verbosity_error()
file_handler = logging.FileHandler('logs.txt')
logger.addHandler(file_handler)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



def ngrams(sequence: List[str], n: int) -> Iterable:
    """
    Directly taken from nltk package to avoid dependency.

    Parameters
    ----------
    sequence : list
        The sequence of items to be n-grammed.
    n : int
        The order of the n-grams to be extracted.

    Returns
    -------
    Iterable
        The n-grams generated from the sequence.
    """
    iterables = tee(sequence, n)
    for i, sub_iterable in enumerate(iterables):
        for _ in range(i):
            next(sub_iterable, None)
    return zip(*iterables)


def sha1_hash32(data):
    """
    Directly taken from datasketch package to avoid dependency.

    Parameters
    ----------
    data : bytes

    Returns
    -------
    int
    """
    return struct.unpack("<I", hashlib.sha1(data).digest()[:4])[0]


def embed_func(
    content: str,
    idx: int,
    *,
    num_perm: int,
    ngram_size: int,
    hashranges: List[Tuple[int, int]],
    permutations: np.ndarray,
) -> Dict[str, Any]:
    """
    Combined with some datasketch code to avoid dependency.

    Parameters
    ----------
    content : str
        The content to be embedded.
    idx : int
        The index of the content.
    num_perm : int
        The number of permutations.
    ngram_size : int
        The size of n-grams.
    hashranges : List[Tuple[int, int]]
        The ranges of hash values.
    permutations : np.ndarray
        The permutations for the minhash.

    Returns
    -------
    Dict[str, Any]
        The hash values in each range and the index.
    """
    hashvalues = np.ones(num_perm, dtype=np.uint64) * MAX_HASH
    tokens = content.split(" ")
    tokens = [
            str.encode(" ".join(tokens[i : i + ngram_size]))
            for i in range(len(tokens) - ngram_size)
    ]
    hv = np.array([sha1_hash32(token) for token in tokens], dtype=np.uint64)  # noqa: E501
    a, b = permutations
    phv = np.bitwise_and(((hv * np.tile(a, (len(hv), 1)).T).T + b) % MERSENNE_PRIME, MAX_HASH)  # noqa: E501
    hashvalues = np.vstack([phv, hashvalues]).min(axis=0)
    Hs = [bytes(hashvalues[start:end].byteswap().data) for start, end in hashranges]
    return {"__signatures__": Hs, "__id__": idx}


def optimal_param(
    threshold: float,
    num_perm: int,
    false_positive_weight: float = 0.5,
    false_negative_weight: float = 0.5,
):
    """
    Compute the optimal `MinHashLSH` parameter that minimizes the weighted sum
    of probabilities of false positive and false negative, taken from datasketch.

    Parameters
    ----------
    threshold : float
        The threshold for similarity.
    num_perm : int
        The number of permutations.
    false_positive_weight : float
        The weight of false positive.
    false_negative_weight : float
        The weight of false negative.

    Returns
    -------
    Tuple[int, int]
        The optimal `b` and `r` parameters.
    """

    def false_positive_probability(threshold: float, b: int, r: int):
        """Source: `datasketch.lsh`"""

        def proba(s):
            return 1 - (1 - s ** float(r)) ** float(b)

        a, _ = integrate(proba, 0.0, threshold)
        return a

    def false_negative_probability(threshold: float, b: int, r: int):
        """Source: `datasketch.lsh`"""

        def proba(s):
            return 1 - (1 - (1 - s ** float(r)) ** float(b))

        a, _ = integrate(proba, threshold, 1.0)
        return a

    min_error = float("inf")
    opt = (0, 0)
    for b in range(1, num_perm + 1):
        max_r = int(num_perm / b)
        for r in range(1, max_r + 1):
            fp = false_positive_probability(threshold, b, r)
            fn = false_negative_probability(threshold, b, r)
            error = fp * false_positive_weight + fn * false_negative_weight
            if error < min_error:
                min_error = error
                opt = (b, r)
    return opt


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
    def run():
        from datasets import load_dataset, load_from_disk, concatenate_datasets
        import json
        global uf
        column = 'text'
        ngram_size = 3
        threshold = 0.85
        output = 'group_deduped_dataset'
        num_perm = 256
        ROOT_PATH = "pilev2_local_dedup"
        OUTPUT_BASE = Path(output or "output")
        OUTPUT_BASE.mkdir(exist_ok=True, parents=True)
        output = OUTPUT_BASE


        time_measures = {}
        start_time = time.time()

        B, R = optimal_param(threshold, num_perm)
        HASH_RANGES = [(i * R, (i + 1) * R) for i in range(B)]
        HASH_TABLES = [defaultdict(set) for _ in range(B)]


        with open ("temp.json", "r") as f:
            data = json.load(f)

        in_common = data.pop("common_group")
        for group_name in ["group_1", "group_2"]:
            group = []
            logger.info("Read datasets in group")
            for name in data[group_name]:
                logger.info(f"Reading {name} dataset")
                ds_path = os.path.join(ROOT_PATH, name)
                ds = load_from_disk(ds_path)
                logger.info(f"Dataset info: {ds}")
                if "check_char_repetition_criteria" in ds.column_names and "check_flagged_words_criteria" in ds.column_names and "check_stop_word_ratio_criteria" in ds.column_names:
                    ds = ds.remove_columns(
                        [
                            'check_char_repetition_criteria',
                            'check_flagged_words_criteria',
                            'check_stop_word_ratio_criteria'
                        ]
                    )
                group.append(ds)
            logger.info(f"Reading C4 datasets")
            for name in in_common:
                logger.info(f"Reading {name} dataset")
                ds_path = os.path.join(ROOT_PATH, name)
                ds = load_from_disk(ds_path)
                logger.info(f"Dataset info: {ds}")
                if "check_char_repetition_criteria" in ds.column_names and "check_flagged_words_criteria" in ds.column_names and "check_stop_word_ratio_criteria" in ds.column_names:
                    ds = ds.remove_columns(
                        [
                            'check_char_repetition_criteria',
                            'check_flagged_words_criteria',
                            'check_stop_word_ratio_criteria'
                        ]
                    )
                group.append(ds)
            logger.info(f"Starting to concat datasets")
            ds = concatenate_datasets(group)
            ds = ds.remove_columns(["id"])
            ds = ds.add_column("id", range(len(ds)))
            logger.info(f"{group_name} dataset: {ds}" )
            DATA_SIZE = len(ds)
            PERMUTATIONS = np.array(
                [
                    (
                        RNG.randint(1, MERSENNE_PRIME, dtype=np.uint64),
                        RNG.randint(0, MERSENNE_PRIME, dtype=np.uint64),
                    )
                    for _ in range(num_perm)
                ],
                dtype=np.uint64,
            ).T

            time_measures["minhash"] = time.time()
            embedded = ds.map(
                function=embed_func,
                fn_kwargs={
                    "num_perm": num_perm,
                    "hashranges": HASH_RANGES,
                    "ngram_size": ngram_size,
                    "permutations": PERMUTATIONS,
                },
                input_columns=[column],
                remove_columns=ds.column_names,
                num_proc=os.cpu_count(),
                with_indices=True,
                desc="Fingerprinting...",
            )
            time_measures["minhash"] = time.time() - time_measures["minhash"]

            time_measures["clustering"] = time.time()
            batch_size: int = 10000
            for i in tqdm(
                range(0, len(embedded), batch_size), dynamic_ncols=True, desc="Iterating MinHashes..."  # noqa: E501
            ):
                batch = embedded[i : i + batch_size]
                for key, Hs in zip(batch["__id__"], batch["__signatures__"]):
                    for H, hashtable in zip(Hs, HASH_TABLES):
                        hashtable[H].add(key)
            for table in tqdm(HASH_TABLES, dynamic_ncols=True, desc="Clustering..."):
                for cluster in table.values():
                    if len(cluster) <= 1:
                        continue
                    idx = min(cluster)
                    for x in cluster:
                        uf.union(x, idx)
            time_measures["clustering"] = time.time() - time_measures["clustering"]

            time_measures["filtering"] = time.time()
            gc.freeze()
            gc.disable()
            ds = ds.map(
                function=lambda _, idx: {"__cluster__": uf.find(idx)},
                with_indices=True,
                num_proc=os.cpu_count(),
                new_fingerprint=str(random.getrandbits(128)),
                desc="Finding clusters...",
            )
            gc.enable()
            gc.collect()
            # This is where the deduplication happens
            # Since there is no easy groupby in datasets
            # I will use this simple filter for now
            final_data = ds.filter(
                function=lambda record, idx: record["__cluster__"] == idx,
                with_indices=True,
                num_proc=os.cpu_count(),
                desc="Filtering clusters...",
            )
            time_measures["filtering"] = time.time() - time_measures["filtering"]

            time_measures["save"] = time.time()
            final_data = final_data.remove_columns(["__cluster__"])
            final_data.save_to_disk(f"{output}/{group_name}")
            time_measures["save"] = time.time() - time_measures["save"]

            FINAL_DATA_SIZE = len(final_data)
            DUP_SIZE = DATA_SIZE - FINAL_DATA_SIZE
            PAD = 32

            for key, value in time_measures.items():
                logger.info(f"{key:<{PAD}}: {value:.2f} seconds")
            logger.info(f"{'Data Number (before)':<{PAD}}: {DATA_SIZE}")
            logger.info(
                f"{'Data Number (after)':<{PAD}}: {FINAL_DATA_SIZE} ({FINAL_DATA_SIZE / DATA_SIZE:.2%})"  # noqa: E501
            )
            logger.info(f"{'Duplicate Number':<{PAD}}: {DUP_SIZE} ({DUP_SIZE / DATA_SIZE:.2%})")  # noqa: E501
            logger.info(f"{'Total Time':<{PAD}}: {time.time() - start_time:.2f} seconds")
            logger.info(f"{'Deduplicated Dataset':<{PAD}}: {output}")
            logger.info("🤗 Happy Deduplicating 🤗")
    mp.set_start_method("fork", force=True)
    uf = UnionFind()
    typer.run(run)
