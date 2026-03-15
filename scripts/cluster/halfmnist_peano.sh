#!/bin/bash
#SBATCH --job-name=halfmnist_peano
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --output=logs/halfmnist_peano_%j.out
#SBATCH --error=logs/halfmnist_peano_%j.err
#SBATCH --partition=edu-long
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

# Run the script
python -m experiments.run_halfmnist_peano --preset peano_medium_v1
