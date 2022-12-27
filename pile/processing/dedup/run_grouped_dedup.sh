#! /bin//bash

DATA_PATH=~/pilev2_data/pilev2_local_dedup/
GROUPED_DUPS=./groups.json
OUTPUT_PATH=./deduped/

python ./grouped_dup.py --dataset_path $DATA_PATH --grouped_dup_path $GROUPED_DUPS --output_path $OUTPUT_PATH