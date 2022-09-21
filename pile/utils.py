import os
from typing import Union, List
from pathlib import Path
from best_download import download_file
import logging
import shutil
import random
from diskcache import Cache
import pyfra as pf
import hashlib
from tqdm import tqdm

logger = logging.getLogger(__name__)


def directory_size(path: Union[str, Path]) -> int:
    """
    Calculate the size of a directory and all subdirectories on disk.
    """
    path = str(path)
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    return total_size


def schema_from_examples(examples: List[dict]) -> dict:
    """
    Returns the schema of the given examples.
    """
    import genson

    builder = genson.SchemaBuilder(schema_uri=None)
    for example in examples:
        builder.add_object(example)
    return builder.to_schema()


def generate_examples(dataset, out_path, n=5, shuffle=True):
    """Generate a small number of examples from the dataset.

    Args:
        dataset: The dataset to generate examples from.
        out_path: The path to write the examples to (JSONL).
        n: The number of examples to generate.
        shuffle: Whether to shuffle the dataset - requires the whole dataset to be in memory.
    """
    from .file_utils import dump_jsonl

    if shuffle:
        documents = list(dataset.documents())
        random.shuffle(documents)
        examples = documents[:n]
    else:
        examples = []
        for i, doc in enumerate(dataset.documents()):
            examples.append(doc)
            if i >= n:
                break
    dump_jsonl(examples, out_path)
    return examples


def pile_cache_dir() -> Path:
    """Returns the path to the global cache directory for all datasets.

    Can be overridden by setting the environment variable
    PILE_CACHE_DIR.
    """
    return Path(os.getenv("PILE_CACHE_DIR", Path.home() / "pile_datasets"))


def touch(filename: Union[str, Path]):
    """
    Creates an empty file at the given path.
    """
    Path(filename).touch()


def done_path(path):
    """
    Path to the hidden file that indicates that a component has been downloaded
    and verified.
    """
    return Path(path).parent / ("." + Path(path).name + ".done")


def component_exists(path: Union[str, Path]) -> bool:
    """
    Returns true if the given path exists, as well as the `path + ".done"`
    file, which indicates that the component has been downloaded and verified.
    """
    return Path(path).exists() and done_path(path).exists()


def mark_done(path: Union[str, Path]):
    touch(done_path(path))


def download(
    url: str,
    out_path: Union[str, Path],
    mirrors: List[Union[str, Path]] = None,
    checksum=None,
    force=False,
):
    """Downloads a file to out_path, and optionally checks its sha_256
    checksum.

    If any mirrors are provided, it will attempt to download from each of these if
    the initial url fails.

    Args:
        url: The url to download from.
        out_path: The path to download to.
        mirrors: A list of urls to try if the initial url fails.
        checksum: The sha256 hash of the file to download. If provided, the file will be
                 downloaded and verified.
    """
    # early exit if already downloaded:
    if component_exists(out_path) and not force:
        return

    # make parent directories if necessary:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    # download the file:
    try:
        success = download_file(url, str(out_path), expected_checksum=checksum)
    except (SystemExit, KeyboardInterrupt):
        raise
    except:
        print("here")
        import traceback

        traceback.print_exc()
        error_msg = f"Failed to download {url} to {out_path}"
        if mirrors is not None:
            error_msg += f"\nTrying mirror: {mirrors[0]}"
        logger.error(error_msg)
        if mirrors is not None:
            # recursively try remaining mirrors:
            next_mirror = mirrors.pop(0)
            return download(
                url=next_mirror,
                out_path=out_path,
                remaining_mirrors=None if len(mirrors) == 0 else mirrors,
                checksum=checksum,
            )
        else:
            error_msg += "\nNo more mirrors to try."
            raise Exception(error_msg)

    # mark the file as done:
    if success:
        mark_done(out_path)
    return


def utf8len(s: str) -> int:
    """
    Return the number of UTF-8 characters in a string.
    """
    return len(s.encode("utf-8"))


def sh(x):
    return pf.sh(x)


def rm_if_exists(path):
    try:
        if os.path.exists(path):
            shutil.rmtree(path)
    except NotADirectoryError:
        os.remove(path)


def _disk_cache():
    cache_dir = pile_cache_dir()
    disk_cache = Cache(cache_dir)
    return disk_cache


cache = _disk_cache()
disk_cache = cache.memoize  # caches the result of a function call to disk


def sha256sum(filename, expected=None):
    h = hashlib.sha256()
    b = bytearray(128 * 1024)
    mv = memoryview(b)
    progress = tqdm(total=os.path.getsize(filename), unit="byte", unit_scale=1)
    tqdm.write(f"Verifying checksum for {filename}")
    with open(filename, "rb", buffering=0) as f:
        for n in iter(lambda: f.readinto(mv), 0):
            h.update(mv[:n])
            progress.update(n)
    progress.close()
    checksum = h.hexdigest()
    if expected:
        assert checksum == expected
        print("CHECKSUM OK", filename)
    else:
        print(filename, checksum)
    return checksum
