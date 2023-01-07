import argparse
import re
import yaml

from datasets import disable_caching, load_from_disk
from pathlib import Path
disable_caching()

def get_yaml_contents(text):
  # Remove any non-ascii characters and x0007
  text = re.sub(r'\x07',r'', text)
  text = re.sub(r'[^\x00-\x7f]',r'', text)
  # remove single quotes
  text = re.sub(r"'",r'', text)
  lines = text.split("\n")
  yaml_contents = []
  in_yaml = False
  start, end = 0, 0
  for idx, line in enumerate(lines):
    if line == "---":
      if in_yaml:
        end = idx
        break
      else:
        in_yaml = True
        start = idx
        continue
    if in_yaml:
      yaml_contents.append(line)

  return yaml.safe_load("\n".join(yaml_contents)), (start, end)

def reformatter(example):
  yaml_contents, (start, end) = get_yaml_contents(example["text"])

  # remove the yaml contents from the text
  lines = example["text"].split("\n")
  lines = lines[:start] + lines[end+1:]
  # add the title and abstract
  if "title" in yaml_contents:
    lines = [f"{yaml_contents['title']}\n"] + lines
  if "abstract" in yaml_contents:
    lines = [f"{yaml_contents['abstract']}\n"] + lines

  example["text"] = "\n".join(lines)
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
arxiv_ds = load_from_disk(data_dir)
arxiv_ds = arxiv_ds.map(reformatter)

arxiv_ds.save_to_disk(output_dir)

# python fix_format_arxiv.py --data_dir /fsx/home-nathan/work/pilev2/pile/processing/fix_format/pile-v2-eda/local_dedup/arXiv_ver2 --output_dir /fsx/home-nathan/work/pilev2/pile/processing/fix_format/pile-v2-eda/reformated/arXiv_ver2
# /fsx/home-nathan/work/pilev2/pile/processing/fix_format/pile-v2-eda/local_dedup/arXiv_ver2
# /fsx/home-nathan/work/pilev2/pile/processing/fix_format/pile-v2-eda/reformated/arXiv_ver2