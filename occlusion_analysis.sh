#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G

module load Anaconda3/2022.10
source activate pytorch1.11cu113
cd /mnt/parscratch/users/acq22ja/learning-from-brains
python3 scripts/occlusion.py \
--model-dir results/models/downstream/ds002105/LinearBaseline_train-decoding_2023-08-27_02-26-34 \
--data data/downstream/ds002105 \
--seed 1234 \
--architecture LinearBaseline \
--training-style "decoding" \
--n-train-subjects-per-dataset 1 \
--n-val-subjects-per-dataset 1 \
--n-test-subjects-per-dataset 9 \
--test-steps 1000 \
--decoding-target "task_label.pyd" \
--num-decoding-classes 26 \
--training-steps 1000 \
--per-device-training-batch-size 64 \
--learning-rate 1e-4 \
--log-dir "results/models/downstream/ds002105" \
--fp16 True \
--log-every-n-steps 1000
