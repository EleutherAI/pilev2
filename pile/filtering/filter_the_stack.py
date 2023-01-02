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
import IPython.nbconvert as nbconvert


logging.basicConfig(
    level = logging.INFO,
    format=  '%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename=f"pile/logs/the_stack_{time.time()}.log")
logger = logging.getLogger(__name__)


FLAG_LANG_LIST = ["HTML","JSON","JSON5","JSONLD","JSONiq","Java Server Pages","YAML","XML","HTTP","CSV","SVG","Cython","Diff","Groff","Groovy Server Pages","INI","Inno Setup","LabVIEW","Less","Logos","MTML","Modelica","Module Management System","NSIS","Nginx","NumPy","ObjDump","Parrot Internal Representation","Protocol Buffer","Pure Data","Python traceback","RHTML","SCSS","Sass","ShellSession","Unity3D Asset","XPages","Xojo","desktop"]
FLAG_EXT_FLAT = [".aux",".mkiv",".mkvi",".sty",".toc"]

def filter_flat_langs(example:dict)->bool:
    if "lang" in example and "ext" in example:
        if example["lang"] not in FLAG_LANG_LIST and example["ext"] not in FLAG_EXT_FLAT:
            return True
        else:
            return False
    else:
        return False

def convert_to_py(example:dict):
    content = example["content"]

    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--num_workers", type=int, default=128)
    args = parser.parse_args()

    logger.info("Loading dataset")

    dataset = load_from_disk(args.input_dir)
    logger.info("Loaded dataset")

    logger.info("Filtering flat langs")
    dataset = dataset.filter(filter_flat_langs, num_proc=args.num_workers)
    logger.info("Filtered flat langs")

    logger.info("Saving dataset")
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(args.output_dir)
    logger.info(f"Saved dataset at {args.output_dir}")
