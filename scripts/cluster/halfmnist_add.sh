#!/bin/bash
#SBATCH --job-name=halfmnist_add
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --output=logs/halfmnist_add_%j.out
#SBATCH --error=logs/halfmnist_add_%j.err
#SBATCH --partition=edu-medium
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

# Run the script
python -m experiments.run_halfmnist_supervised --preset add_medium_v1
