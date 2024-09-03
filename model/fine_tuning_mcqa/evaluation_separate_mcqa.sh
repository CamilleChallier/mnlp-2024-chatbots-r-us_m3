#!/bin/bash -l
#SBATCH --chdir /scratch/izar/ckalberm
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=8
#SBATCH --mem 16G
#SBATCH --time 1:00:00
#SBATCH --gres gpu:1
#SBATCH --account cs-552
#SBATCH --qos cs-552
#SBATCH --reservation cs-552

echo STARTING AT `date`
nvidia-smi

cd /home/ckalberm/project-m2-2024-chatbots-r-us
echo SUCCESSFULLY CHANGED LOCATION

python3 -u /home/ckalberm/project-m3-2024-chatbots-r-us/model/fine-tuning/evaluation_separate_mcqa.py

echo FINISHED at `date`