from typing import Union, List
from pathlib import Path
import logging
import json
import zstandard as zstd
import io

logger = logging.getLogger(__name__)


def dump_jsonl(data: List, output_path: Union[str, Path], append: bool = False):
    """
    Write list of objects to a JSON lines file.
    """
    mode = "a+" if append else "w"
    with open(output_path, mode, encoding="utf-8") as f:
        for line in data:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


def stream_jsonl(path: Union[str, Path]):
    """
    Stream JSON lines file.
    """
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def stream_jsonl_zst(path: Union[str, Path]):
    with open(path, "rb") as fh:
        cctx = zstd.ZstdDecompressor()
        reader = io.BufferedReader(cctx.stream_reader(fh))
        for line in reader:
            yield json.loads(line.decode("utf-8"))
