# -*- coding: UTF-8 -*-

import argparse
import json
import logging
import os
import sys
from typing import Dict

import numpy as np
import torch
from nilearn.datasets import fetch_atlas_difumo
from nilearn.regions import signals_to_img_maps
from numpy import random
from torch import manual_seed
from torch.utils.data import Dataset

script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, f'{script_path}/../')
from train import make_model

sys.path.insert(0, f'{script_path}/../../')

from src.batcher import make_batcher
from src.tools import plot_brain_map

__author__ = "jadikesavan1@sheffield.ac.uk"
LOGGER = logging.getLogger(name=__name__)
logging.basicConfig(level=logging.INFO)


def identify_roi(config: Dict = None) -> None:
    """Script's main function; computes brain map for
    reconstruction error of given upstream model in upstream
    validation data"""

    if config is None:
        config = vars(get_args().parse_args())

    random.seed(config["seed"])
    manual_seed(config["seed"])
    os.makedirs(
        config['error_brainmaps_dir'],
        exist_ok=True
    )
    path_model_config = os.path.join(
        config["model_dir"],
        'train_config.json'
    )
    path_tarfile_paths_split = os.path.join(
        config["model_dir"],
        'tarfile_paths_split.json'
    )
    path_pretrained_model = os.path.join(
        config["model_dir"],
        'model_final',
        "pytorch_model.bin"
    )

    enc_path_model_config = os.path.join(
        config["model_dir"],
        'train_config.json'
    )
    enc_path_tarfile_paths_split = os.path.join(
        config["model_dir"],
        'tarfile_paths_split.json'
    )
    enc_path_pretrained_model = os.path.join(
        config["model_dir"],
        'model_final',
        "pytorch_model.bin"
    )

    assert os.path.isfile(path_model_config), \
        f'{path_model_config} does not exist'
    assert os.path.isfile(path_tarfile_paths_split), \
        f'{path_tarfile_paths_split} does not exist'
    assert os.path.isfile(path_pretrained_model), \
        f'{path_pretrained_model} does not exist'

    assert os.path.isfile(enc_path_model_config), \
        f'{enc_path_model_config} does not exist'
    assert os.path.isfile(enc_path_tarfile_paths_split), \
        f'{enc_path_tarfile_paths_split} does not exist'
    assert os.path.isfile(enc_path_pretrained_model), \
        f'{enc_path_pretrained_model} does not exist'

    with open(path_tarfile_paths_split, 'r') as f:
        tarfile_paths_split = json.load(f)

    with open(path_model_config, 'r') as f:
        model_config = json.load(f)

    with open(enc_path_tarfile_paths_split, 'r') as f:
        enc_tarfile_paths_split = json.load(f)

    with open(enc_path_model_config, 'r') as f:
        enc_model_config = json.load(f)

    model_config['pretrained_model'] = path_pretrained_model
    enc_model_config['pretrained_model'] = enc_path_pretrained_model

    path_error_map = os.path.join(
        config['error_brainmaps_dir'],
        f'mean_eval_error_{model_config["training_style"]}.nii.gz'
    )

    # Get the downstream dataset
    batcher = make_batcher(
        training_style=model_config["training_style"],
        sample_random_seq=model_config["sample_random_seq"],
        seq_min=model_config["seq_min"],
        seq_max=model_config["seq_max"],
        bert_seq_gap_min=model_config["bert_seq_gap_min"],
        bert_seq_gap_max=model_config["bert_seq_gap_max"],
        decoding_target=model_config["decoding_target"],
        bold_dummy_mode=model_config["bold_dummy_mode"]
    )

    validation_dataset = batcher.dataset(
        tarfiles=[
            os.path.join(
                config["data_dir"],
                f.split('downstream/')[-1],
            )
            for f in tarfile_paths_split['validation']
        ],
        length=config["n_eval_samples"]
    )
    eval_dataloader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=1
    )

    # Get the trained downstream model
    model = make_model(model_config=model_config)
    weights = np.zeros(1024)

    # Plot the weights on the brain atlas
    difumo = fetch_atlas_difumo(
        dimension=1024,
        resolution_mm=2
    )
    images = [validation_dataset[i] for i in range(0, len(validation_dataset))]

    for batch_i, batch in enumerate(eval_dataloader):
        LOGGER.info(
            f'\tprocessing sample {batch_i} / {config["n_eval_samples"]}'
        )
        img_name = batch["__key__"][0][0]
        batch = {k: v[0] for k, v in batch.items() if '__key__' != k}

        logits, ret_batch = model.forward(batch, return_batch=True)
        unembedded_data = model.unembedder.forward(ret_batch["inputs_embeds"])["outputs"]

        for i in range(0, ret_batch["inputs"][0].shape[0]):
            par_dir = os.path.join(config['error_brainmaps_dir'], img_name)

            if not os.path.isdir(par_dir):
                os.mkdir(par_dir)

            if ret_batch["attention_mask"][0][i] == torch.tensor(1):
                error_img_map = signals_to_img_maps(
                    region_signals=unembedded_data[0][i].detach().cpu().numpy(),
                    maps_img=difumo.maps
                )
                error_img_map.to_filename(os.path.join(par_dir, img_name) + f"_{str(i)}")

                try:
                    plot_brain_map(
                        img=error_img_map,
                        path=os.path.join(
                            par_dir,
                            f'mean_eval_error_{model_config["training_style"]}_{img_name}_{str(i)}.png'
                        ),
                        vmin=np.min(error_img_map.get_fdata()),
                        vmax=0.0035,
                    )
                except Exception as exc:
                    continue


    return None


def get_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='compute upstream reconstruction error brain map for given model; '
                    'as shown in appendix figure 2 of the manuscript.'
    )

    parser.add_argument(
        '--model-dir',
        metavar='DIR',
        type=str,
        help='path to directory where model is stored '
             'for which reconstruction error brain map '
             'is to be computed.'
    )

    parser.add_argument(
        '--enc-model-dir',
        metavar='DIR',
        type=str,
        help='path to directory where encoder model is stored '
             'for which reconstruction error brain map '
             'is to be computed.'
    )
    parser.add_argument(
        '--error-brainmaps-dir',
        metavar='DIR',
        default='results/brain_maps/l1_error',
        type=str,
        help='directory to which error brain map will be stored '
             '(default: results/brain_maps/l1_error)'
    )
    parser.add_argument(
        '--n-eval-samples',
        metavar='N',
        default=50000,
        type=int,
        help='number of random samples to draw for evaluation '
             '(default: 50000)'
    )
    parser.add_argument(
        '--embedding-dim',
        metavar='INT',
        default=768,
        type=int,
        help='dimension of input embedding '
             '(default: 768)'
    )
    parser.add_argument(
        '--data-dir',
        metavar='DIR',
        type=str,
        default='data/upstream',
        help='path to upstream data directory '
             '(default: data/upstream)'
    )
    parser.add_argument(
        '--training-style',
        metavar='STR',
        default='CSM',
        choices=(
            'CSM',
            'BERT',
            'NetBERT',
            'autoencoder',
            'decoding'
        ),
        type=str,
        help='training framework / style (default: CSM); '
             'one of {BERT, CSM, NetBERT, autoencoder, decoding}'
    )

    parser.add_argument(
        '--parcellation-dim',
        metavar='INT',
        default=1024,
        type=int,
        help='dimension of input data parcellation (default: 1024). '
             '! This is fixed for the current up-/downstream data.'
    )

    parser.add_argument(
        '--seed',
        metavar='INT',
        default=1234,
        type=int,
        help='random seed (default: 1234)'
    )

    return parser


if __name__ == '__main__':
    identify_roi()
    # 1. Train the downstream classifier
    # 2. Feed the downstream data in batches
    #       2.1 Get the weights of each hidden neuron (1024)
    #       2.2 Treat the weights as signals and project onto the difumo brain atlas
    #       2.3 Get the projection as an image
    #       2.4 Save the image onto the disk
    # 3. Repeat for every image

