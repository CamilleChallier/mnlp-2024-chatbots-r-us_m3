#!/bin/bash -l
#SBATCH --chdir /scratch/izar/challier/project-m3-2024-chatbots-r-us/model
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=8
#SBATCH --mem 16G
#SBATCH --time 0:30:00
#SBATCH --gres gpu:1
#SBATCH --account cs-552
#SBATCH --qos cs-552
#SBATCH --reservation cs-552

echo STARTING AT `date`
echo IN `pwd`
nvidia-smi

python3 -u evaluator.py

echo FINISHED at `date`