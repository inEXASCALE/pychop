#!/bin/bash

#SBATCH --job-name=xinye
#SBATCH --nodes=1
#SBATCH --mem=512G
#SBATCH --gpus=a100_7g.80gb:1
#SBATCH --time=2000
#SBATCH --mail-type=ALL
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

module load cuda
python example_CNN_ft.py
