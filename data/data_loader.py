
import os
import sys

import json
import torch
from torch.utils.data import Dataset, DataLoader

top_config = {}
cwd = os.getcwd()  # current working directory
cfp = os.path.dirname(os.path.abspath(__file__))  # current file path
os.chdir(cfp)
root_full_path = os.path.abspath("..")
sys.path.append(root_full_path)
config_full_path = os.path.join(root_full_path, 'config/config.json')
if not os.path.exists(config_full_path):
    print(f"{config_full_path} doesn't exist.")
    exit(0)
with open(config_full_path, 'r') as file:
    top_config = json.load(file)
os.chdir(cwd)

# TODO(diwei): Modify the class' detail including preproess, dataset or remove it.
class TopDataset(Dataset):
    def __init__(self):
        self.num = 10

        self.xs = torch.rand(top_config["model_params"]["source_length"],
                             self.num, top_config["model_params"]["d_model"])
        self.us = torch.rand(top_config["model_params"]["target_length"],
                             self.num, top_config["model_params"]["d_model"])
        self.gts = torch.rand(
            self.num, top_config["model_params"]["num_classes"])

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        return {"x": self.xs[:, idx, :], "u": self.us[:, idx, :], "gt": self.gts[idx, :]}


# TODO(diwei): Encapsulate the dataset and the dataloader into a class
train_dataset = TopDataset()
validation_dataset = TopDataset()
batch_size = top_config["train_params"]["batch_size"]
train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
validation_data_loader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=True)


def preprocess(batch):
    """The function to do preprocess the dataset.

    Parameters
    ----------
    batch : dict
        The data batch including source, target and ground truth from 
        data loader.

    Returns
    -------
    tuple
        The list of the data including source, target and ground truth with
        the batch size as first dimension .
    """
    return batch["x"].permute(1, 0, 2), batch["u"].permute(1, 0, 2),  batch["gt"]
