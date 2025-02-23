#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=32gb
#SBATCH --partition=gpu
#SBATCH --gpus=a100:4
#SBATCH --time=72:00:00
#SBATCH --output=%x.%j.out

STAGE3_PATH=$1
OUTPUT=$2
DATA_FILE=$3

date;hostname;pwd

module load conda
conda activate nvila

bash longvila/train/5_long_sft_frames_ldct.sh $STAGE3_PATH $OUTPUT $DATA_FILE