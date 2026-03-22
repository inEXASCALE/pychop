#!/bin/bash

#SBATCH --job-name=llm_finetune
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=a100_7g.80gb:2
#SBATCH --time=72:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

module load cuda/12.8
module load cudnn/9.5.0_cuda12

source activate myenv
python llm_finetune_mx_nccl.py \
    --model-name-or-path "meta-llama/Llama-2-7b-chat-hf" \
    --dataset-name "Open-Orca/OpenOrca" \
    --block-size 32 \
    --epochs 3 \
    --batch-size 8 \
    --lr 5e-5