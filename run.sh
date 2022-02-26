#!/bin/bash
# SBATCH --time=16:00:00
# SBATCH --account=rrg-ebrahimi
# SBATCH --mem=48G
# SBATCH --gres=gpu:1
# SBATCH --array=1-5
# SBATCH --output=out/%x/%A_%a.out
# SBATCH --cpus-per-task=4
# SBATCH --mail-type=BEGIN
# SBATCH --mail-type=END
# SBATCH --mail-type=FAIL

source /home/shrutij/venvs/bbs/bin/activate
wandb offline

python baselines/dreamerv2/main.py