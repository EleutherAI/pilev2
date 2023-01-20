#!/bin/bash
#SBATCH --job-name=PileV2Reddit2021
#SBATCH --output=./PileV2Reddit2021.o
#SBATCH --error=./PileV2Reddit2021.e
#SBATCH --time=168:00:00
#SBATCH --mem=960GB
#SBATCH --cpus-per-task=128
#SBATCH --partition=cpu128
#SBATCH --comment=carper
#SBATCH --export=ALL
# ===== END SLURM OPTIONS =====
source /fsx/home-erfan/miniconda3/bin/activate pilev2
cd /fsx/home-erfan/pilev2/pile/processing/dedup
mkdir -p job_logs 
DATA_PATH=/fsx/shared/pilev2/deduped_2_2/deduped_2_2/group_2_2
OUTPUT_PATH=/fsx/shared/pilev2_hashes/group_2
dataset=PileV2Reddit2021
python generate_min_hashes.py --dataset-path $DATA_PATH --column text --threshold 0.85 --output $OUTPUT_PATH/$dataset
mv PileV2Reddit2021.o job_logs/PileV2Reddit2021.o
mv PileV2Reddit2021.e job_logs/PileV2Reddit2021.e