#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --comment=GPULinearOpenNeuro
#SBATCH --mem=16G

module load Anaconda3/2022.10
source activate pytorch1.11cu113
cd /mnt/parscratch/users/acq22ja/learning-from-brains/
python3 scripts/train.py \
--data data/downstream/HCP \
--n-train-subjects-per-dataset 48 \
--n-val-subjects-per-dataset 6  \
--n-test-subjects-per-dataset 20 \
--architecture "LinearBaseline" \
--training-style "decoding" \
--decoding-target "task_label.pyd" \
--num-decoding-classes 20 \
--training-steps 50000 \
--per-device-training-batch-size 64 \
--learning-rate 1e-4 \
--log-dir "results/models/downstream/HCP" \
--fp16 True \
--log-every-n-steps 10000 \
--plot-model-graph False
