#!/bin/bash
#SBATCH --job-name=mnist_sum_parity
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --output=logs/mnist_sum_parity_%j.out
#SBATCH --error=logs/mnist_sum_parity_%j.err
#SBATCH --partition=edu-long
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

# Run the script
python -m experiments.run_mnist_sum_parity_supervised --preset biased_tight_v2 "$@"
