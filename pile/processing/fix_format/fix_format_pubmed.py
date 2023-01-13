import argparse
import re
import yaml

from datasets import disable_caching, load_from_disk
from pathlib import Path
disable_caching()

def get_body_text(text):
  lines = text.split("\n")
  body_contents = []
  in_body = False
  start, end = 0, 0
  for idx, line in enumerate(lines):
    if "==== Body" in line:
      in_body = True
      start = idx
      continue
    elif "==== " in line and in_body:
      end = idx
      break
    if in_body:
      body_contents.append(line)

  return "\n".join(body_contents), (start, end)

def reformatter(example):
  text, (start, end) = get_body_text(example["text"])
  example["text"] = text
  return example


# Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_dir",
    type=str,
    required=True,
    help="The directory where the data is stored.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    required=True,
    help="The directory where the output should be stored.",
)
args = parser.parse_args()

# create the output directory if it does not exist
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

data_dir = Path(args.data_dir)
pubmed_ds = load_from_disk(data_dir)
pubmed_ds = pubmed_ds.map(reformatter)
print(pubmed_ds[0]["text"])

pubmed_ds.save_to_disk(output_dir)

# python fix_format_pubmed.py --data_dir /work/pilev2/pile/processing/fix_format/pile-v2-eda/local_dedup/PubMed_ver2 --output_dir /work/pilev2/pile/processing/fix_format/pile-v2-eda/reformatted/PubMed_ver2

# /fsx/home-nathan/work/pilev2/pile/processing/fix_format/pile-v2-eda/local_dedup/arXiv_ver2
# /fsx/home-nathan/work/pilev2/pile/processing/fix_format/pile-v2-eda/reformated/arXiv_ver2