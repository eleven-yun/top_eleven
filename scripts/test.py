import os
import sys
import torch
import torch.nn as nn

cwd = os.getcwd()  # current working directory
cfp = os.path.dirname(os.path.abspath(__file__))  # current file path
os.chdir(cfp)
sys.path.append("..")
from nn_modules.transformer.top_former import TopFormer
os.chdir(cwd)

top_former = TopFormer(num_layers=6, d_model=512, nhead=8, num_classes=3)

# TODO: remove this part
for param in top_former.parameters():
    if param.dim() > 1:
        nn.init.xavier_uniform_(param)

model_file_path = 'top_eleven.pt'

# TODO: remove this line
torch.save(top_former.state_dict(), model_file_path)

top_former.load_state_dict(torch.load(model_file_path))
top_former.eval()

# TODO: replace with real data
x = torch.rand(50, 1, 512)
u = torch.rand(10, 1, 512)

p = top_former(x, u)

# TODO: replace with better interface
print(p)
