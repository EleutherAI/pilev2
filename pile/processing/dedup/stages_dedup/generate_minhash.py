"""
Generate MinHash signatures for each document in the dataset.
It generates a dataset with the following fields:
    - __signatures__: a list of minhash signatures
    - __id__: the id of the document (not used)
The output file has the same name as the input file with the suffix "_minhash" under the same directory.

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


SEED = 42
NON_ALPHA = re.compile("[^A-Za-z_0-9]")
RNG = np.random.RandomState(SEED)
MAX_HASH = np.uint64((1 << 32) - 1)
MERSENNE_PRIME = np.uint64((1 << 61) - 1)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
datasets.logging.set_verbosity_error()


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
    tokens = {" ".join(t) for t in ngrams(NON_ALPHA.split(content), ngram_size)}
    hv = np.array([sha1_hash32(token.encode("utf-8")) for token in tokens], dtype=np.uint64)  # noqa: E501
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

if __name__ == "__main__":
    def run(
        dataset_path: str = typer.Option(..., help="The dataset to use"),  # noqa: E501
        column: str = typer.Option(..., help="Dataset column"),
        ngram_size: int = typer.Option(5, help="The ngram size to use for MinHash"),
        num_perm: int = typer.Option(256, help="Number of permutations"),
        threshold: float = typer.Option(0.7, help="Minhash threshold"),
        output: str = typer.Option(None, help="Store the minhash of the dataset, None to store in the same dir of dataset_path but added 'minhash' as suffix"),
    ):
        # write the output in the same dir of dataset_path but added 'minhash' as suffix
        if output is None:
            # make sure the dataset_path does not end with a slash
            assert(not dataset_path.strip().endswith("/"))
            output = dataset_path + '_minhash'
        typer.echo(f"Output will be stored in {output}")

        logging.basicConfig(level=logging.INFO)

        start_time = time.time()
        time_measures = {}
        B, R = optimal_param(threshold, num_perm)
        HASH_RANGES = [(i * R, (i + 1) * R) for i in range(B)]
        HASH_TABLES = [defaultdict(set) for _ in range(B)]
        time_measures["load_dataset"] = time.time()
        ds = load_from_disk(dataset_path)
        time_measures["load_dataset"] = time.time() - time_measures["load_dataset"]
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
        time_measures["save_dataset"] = time.time()
        embedded.save_to_disk(output)
        time_measures["save_dataset"] = time.time() - time_measures["save_dataset"]

        PAD = 32
        MINHASH_DATA_SIZE = len(embedded)

        for key, value in time_measures.items():
            logger.info(f"{key:<{PAD}}: {value:.2f} seconds")
        logger.info(f"{'Data size':<{PAD}}: {DATA_SIZE}")
        logger.info(
            f"{'Number of minhash generated':<{PAD}}: {MINHASH_DATA_SIZE} "  # noqa: E501
        )
        logger.info(f"{'Total Time':<{PAD}}: {time.time() - start_time:.2f} seconds")
        logger.info(f"{'Minhash dataset file':<{PAD}}: {output}")
        logger.info(f"Finish Generating Minhashes for {dataset_path}")
        logger.info("🤗 Happy Deduplicating 🤗")

    mp.set_start_method("fork", force=True)
    typer.run(run)