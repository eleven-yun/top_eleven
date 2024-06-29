
from utils.config import top_config
from torch.utils.data import Dataset, DataLoader
import torch

import os
import sys

'''
Decouple the reference path.
'''
cwd = os.getcwd()  # current working directory
cfp = os.path.dirname(os.path.abspath(__file__))  # current file path
os.chdir(cfp)
sys.path.append(os.path.abspath(".."))
os.chdir(cwd)

# TODO(diwei): modify
class TopDataset(Dataset):
    def __init__(self):
        self.size = 10

        self.xs = torch.rand(top_config["model_params"]["src_length"],
                             self.size, top_config["model_params"]["d_model"])
        self.us = torch.rand(top_config["model_params"]["enc_length"],
                             self.size, top_config["model_params"]["d_model"])
        self.gts = torch.rand(
            self.size, top_config["model_params"]["num_classes"])

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {"x": self.xs[:, idx, :], "u": self.us[:, idx, :], "gt": self.gts[idx, :]}


# TODO(diwei): encapsulate
train_dataset = TopDataset()
val_dataset = TopDataset()
test_dataset = TopDataset()
train_loader = DataLoader(dataset=train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=4, shuffle=True)
