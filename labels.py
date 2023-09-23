import json
import multiprocessing
import os

import torch
import webdataset as wds

DIRECTORY = "/mnt/parscratch/users/acq22ja/learning-from-brains/data/downstream/HCP"
LABEL_DICT_PATH = "/mnt/parscratch/users/acq22ja/learning-from-brains/data/downstream/HCP_labels.json"

if __name__ == '__main__':

    label_dict = {}
    os.chdir(DIRECTORY)
    for file in os.listdir(DIRECTORY):
        if ".tar" in file:
            dataset = wds.WebDataset(file).decode("pil")
            dataloader = torch.utils.data.DataLoader(dataset, num_workers=multiprocessing.cpu_count(), batch_size=1)
            for inputs in dataloader:
                for item in inputs["task_name"]:
                    if inputs["__key__"][0] not in label_dict.keys():
                        label_dict[inputs["__key__"][0]] = item.decode("UTF-8")
                    else:
                        print(f"{label_dict[inputs['__key__'][0]]} is repeating")

    with open(LABEL_DICT_PATH, "w") as fp:
        json.dump(obj=label_dict, fp=fp)

