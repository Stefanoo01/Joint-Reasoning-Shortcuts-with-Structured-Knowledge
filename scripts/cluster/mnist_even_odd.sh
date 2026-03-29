#!/bin/bash
#SBATCH --job-name=mnist_even_odd
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --output=logs/mnist_even_odd_%j.out
#SBATCH --error=logs/mnist_even_odd_%j.err
#SBATCH --partition=edu-long
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

# Run the script
python -m experiments.run_mnist_even_odd_supervised --preset add_sum_relaxed_v1 --lam2 0.4
