#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

module load Anaconda3/2022.10
source activate pytorch1.11cu113
cd learning-from-brains
python3 scripts/projection.py \
--model-dir results/models/downstream/ds002105/GPT_lrs-4_hds-12_embd-768_train-decoding_lr-0001_bs-64_drp-01_2023-08-26_19-26-45/ \
--n-eval-samples 50 \
--data-dir data/downstream \
--error-brainmaps-dir results/brain_maps/xai_projection_new \
--seed 1234
