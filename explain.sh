#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

module load Anaconda3/2022.10
source activate pytorch1.11cu113
cd /mnt/parscratch/users/acq22ja/learning-from-brains
python3 scripts/analyses/explainability.py \
--model-dir results/models/upstream/GPT_lrs-4_hds-12_embd-768_train-CSM_lr-0005_bs-192_drp-01 \
--n-eval-samples 1000 \
--data data/downstream/HCP \
--seed 1234 \
--architecture GPT \
--n-train-subjects-per-dataset 11 \
--n-val-subjects-per-dataset 3 \
--n-test-subjects-per-dataset 9 \
--decoding-target "task_label.pyd" \
--num-decoding-classes 26 \
--training-steps 10000 \
--per-device-training-batch-size 64 \
--learning-rate 1e-4 \
--log-dir "results/models/downstream/ds002105" \
--fp16 True \
--log-every-n-steps 1000 \
--plot-model-graph True
