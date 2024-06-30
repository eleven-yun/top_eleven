from torch.optim import Adam
from torch import nn, optim
import torch
import math
import time
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

from nn_modules.transformer.top_former import TopFormer
from utils.config import top_config
from data.data_loader import train_loader, val_loader

model = TopFormer(num_layers=top_config["model_params"]["num_layers"], d_model=top_config["model_params"]
                  ["d_model"], nhead=top_config["model_params"]["nhead"], num_classes=top_config["model_params"]["num_classes"])

# TODO(diwei): remove


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# TODO(diwei): remove
def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform(m.weight.data)


# TODO(diwei): remove
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


# TODO(diwei): remove
print(f'The model has {count_parameters(model):,} trainable parameters')
model.apply(initialize_weights)


# TODO(diwei): adjust
optimizer = Adam(params=model.parameters(),
                 lr=top_config["opt_params"]["lr"],
                 weight_decay=top_config["opt_params"]["weight_decay"],
                 eps=top_config["opt_params"]["eps"])

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                 verbose=True)

# TODO(diwei): adjust
criterion = nn.CrossEntropyLoss()


def train(model, loader, optimizer, criterion, device='cpu'):
    """THe interface to train the model using training data

    Parameters
    ----------
    model : nn.Module
        The TopFormer model used to train.
    loader : DataLoader
        The loader to visit the data.
    optimizer : Adam
        Optimizer to iterate the params.
    criterion : Loss nn.CrossEntropyLoss
        Loss criterion.
    device : str, optional
        Device used to evaluate, by default 'cpu'

    Returns
    -------
    float
        Average loss.
    """
    epoch_loss = 0
    model.to(device)
    for i, batch in enumerate(loader):
        x = batch["x"]
        u = batch["u"]
        gt = batch["gt"]

        optimizer.zero_grad()
        output = model(x.permute(1, 0, 2), u.permute(1, 0, 2))
        loss = criterion(torch.max(output, dim=0)[0], gt)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        # TODO(diwei): use other progressbar
        print('step :', round((i / (len(loader) + + sys.float_info.epsilon)) * 100, 2),
              '% , loss :', loss.item())

    return epoch_loss / (len(loader) + sys.float_info.epsilon)


def evaluate(model, loader, criterion, device='cpu'):
    """The interface to evaluate the model using validation data

    Parameters
    ----------
    model : nn.Module
        The TopFormer model used to evaluate.
    loader : DataLoader
        The loader to visit the data.
    criterion : Loss nn.CrossEntropyLoss
        Loss criterion.
    device : str, optional
        Device used to evaluate, by default 'cpu'.

    Returns
    -------
    float
        Average loss.
    """
    epoch_loss = 0
    model.to(device)
    with torch.no_grad():
        for _, batch in enumerate(loader):
            x = batch["x"]
            u = batch["u"]
            gt = batch["gt"]
            output = model(x.permute(1, 0, 2), u.permute(1, 0, 2))

            loss = criterion(torch.max(output, dim=0)[0], gt)
            epoch_loss += loss.item()

    return epoch_loss / len(loader)


def run(total_epoch: int, best_loss: float):
    """The interface to run the train process

    Parameters
    ----------
    total_epoch : int
        Total epoch num.
    best_loss : float
        Record the best loss.
    """
    train_losses, test_losses = [], []
    for step in range(total_epoch):
        start_time = time.time()
        train_loss = train(model, train_loader, optimizer, criterion)
        valid_loss = evaluate(model, val_loader, criterion)
        end_time = time.time()

        if step > top_config["train_params"]["warmup"]:
            scheduler.step(valid_loss)

        train_losses.append(train_loss)
        test_losses.append(valid_loss)
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_loss:
            best_loss = valid_loss
            os.makedirs(top_config["train_params"]["save_path"], exist_ok=True)
            torch.save(model.state_dict(),
                       'saved/top_eleven_{0}.pt'.format(valid_loss))

        print(f'Epoch: {step + 1} | Time: {epoch_mins}m {epoch_secs}s')
        print(
            f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(
            f'\tVal Loss: {valid_loss:.3f} |  Val PPL: {math.exp(valid_loss):7.3f}')


if __name__ == '__main__':
    run(total_epoch=top_config["train_params"]
        ["epoch"], best_loss=float('inf'))
