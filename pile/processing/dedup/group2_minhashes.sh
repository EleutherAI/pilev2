#! /bin/bash

# Set variables
DATA_PATH=/fsx/shared/pilev2_local_deduped
GROUPED_DUPS=./groups.json
OUTPUT_PATH=/fsx/shared/pilev2_hashes/group_2

# Read groups.json
groups=$(jq -r '.group_2[]' $GROUPED_DUPS)
common_group=$(jq -r '.common_group[]' $GROUPED_DUPS)


# Concatenate groups and common group
all_groups="$groups $common_group"

# Loop through groups and submit slurm job for each dataset
for dataset in $all_groups; do
    # if $dataset has /shared/ then set $dataset to be the entire path
    if [[ $dataset == *shared* ]]; then
        # DATA_PATH is dataset path without the dataset name
        DATA_PATH=$(dirname $dataset)
        dataset=$(basename $dataset)
        mem=768GB
        cpus=128
        partition=cpu128
        # deduped_reddit_2021

    
    # if $dataset starts with PileV2 or contains C4 then append non_local_dedup to $DATA_PATH
    elif [[ $dataset == PileV2* ]]; then
        DATA_PATH=/fsx/shared/pilev2_local_deduped/non_local_dedup
        mem=768GB
        cpus=128
        partition=cpu128
    elif  [[ $dataset == *C4* ]]; then

        DATA_PATH=/fsx/shared/pilev2_local_deduped
        mem=768GB
        cpus=128
        partition=cpu128

    else
        DATA_PATH=/fsx/shared/pilev2_local_deduped
        mem=128GB
        cpus=32
        partition=cpu64
    fi
    temp_sbatch=./temp_sbatch.slurm
    basename_dataset=$(basename $dataset)
    cat << HELP > $temp_sbatch
#!/bin/bash
#SBATCH --job-name=$basename_dataset
#SBATCH --output=./logs/$basename_dataset.o
#SBATCH --error=./logs/$basename_dataset.e
#SBATCH --time=168:00:00
#SBATCH --mem=$mem
#SBATCH --cpus-per-task=$cpus
#SBATCH --partition=$partition
#SBATCH --comment=carper
#SBATCH --export=ALL
# ===== END SLURM OPTIONS =====
source /fsx/home-erfan/miniconda3/bin/activate pilev2
cd /fsx/home-erfan/pilev2/pile/processing/dedup
python generate_min_hashes.py --dataset-path $DATA_PATH/$dataset --column text --threshold 0.85 --output $OUTPUT_PATH/$dataset

HELP
    sbatch $temp_sbatch
    rm $temp_sbatch
done
