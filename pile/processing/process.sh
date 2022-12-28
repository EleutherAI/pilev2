#! /bin/bash

# python processing.py \
#     --data_dir /fsx/shared/hf_data_pilev2_by_cats/ \
#     --stats_path /fsx/shared/hf_data_pilev2_small_text/stats_dict.pkl
#     --output_dir /fsx/shared/processed_pilev2

python test_datasets.py \
    --data_dir /fsx/shared/pilev2_local_deduped/ \
    --output_dir $(pwd)/test_datasets \