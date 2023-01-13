#! /bin/bash

<<<<<<< HEAD
python processing_compression.py \
    --data_dir /home/nathan/pilev2_group1 \
    --output_dir /home/nathan/pilev2_group1_compress_filtered
=======
# python processing.py \
#     --data_dir /fsx/shared/hf_data_pilev2_by_cats/ \
#     --stats_path /fsx/shared/hf_data_pilev2_small_text/stats_dict.pkl
#     --output_dir /fsx/shared/processed_pilev2

python test_datasets.py \
    --data_dir /fsx/shared/pilev2_local_deduped/ \
    --output_dir $(pwd)/test_datasets \

python processing_compression.py \
    --data_dir /fsx/shared/pilev2_local_deduped/ \
    --output_dir $(pwd)/test_datasets
>>>>>>> 6f1ad901d74a8d295e1cdfd7f70fe0cd9386d04b
