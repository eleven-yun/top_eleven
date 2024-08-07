import math
import time
import os
import sys

import json
import torch
from torch.optim import Adam
from torch import nn, optim

top_config = {}
cwd = os.getcwd()  # current working directory
cfp = os.path.dirname(os.path.abspath(__file__))  # current file path
os.chdir(cfp)
root_full_path = os.path.abspath("..")
sys.path.append(root_full_path)

# import configuration
config_full_path = os.path.join(root_full_path, 'config/config.json')
if not os.path.exists(config_full_path):
    print(f"{config_full_path} doesn't exist.")
    exit(0)
with open(config_full_path, 'r') as file:
    top_config = json.load(file)
os.chdir(cwd)

# import local functions
from nn_modules.transformer.top_former import TopFormer
from data.data_loader import train_data_loader, validation_data_loader, preprocess
from utils.helper import count_parameters, initialize_weights, epoch_time

model = TopFormer(
    num_layers=top_config["model_params"]["num_layers"],
    d_model=top_config["model_params"]["d_model"],
    nhead=top_config["model_params"]["nhead"],
    num_classes=top_config["model_params"]["num_classes"])


# TODO(diwei): Remove this hint
print(f'The model has {count_parameters(model)}: trainable parameters')
model.apply(initialize_weights)


# TODO(diwei): Adjust the type and parameters of the optimizer
optimizer = Adam(params=model.parameters(),
                 lr=top_config["train_params"]["optimize_params"]["lr"],
                 weight_decay=top_config["train_params"]["optimize_params"]["weight_decay"],
                 eps=top_config["train_params"]["optimize_params"]["eps"])

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                 verbose=True)

# TODO(diwei): Adjust the type and parameters of the loss
loss = nn.CrossEntropyLoss()


def train(model, data_loader, optimizer, loss_function, device='cpu'):
    """The interface to train the model using training data.

    Parameters
    ----------
    model : nn.Module
        The model to be trained.
    data_loader : DataLoader
        The interface for accessing training data and its corresponding labels 
        as the ground truth.
    optimizer : Adam
        The optimization algorithm for updating model parameters.
    loss_function : Loss nn.CrossEntropyLoss
        The loss function as objective and constraint for learning model parameters.
    device : str, optional
        The computing device for training the model (default: 'cpu').

    Returns
    -------
    float
        The loss averaged over all training samples in an epoch.
    """
    model.train()
    epoch_loss = 0
    model.to(device)
    for i, batch in enumerate(data_loader):
        x, u, gt = preprocess(batch)

        optimizer.zero_grad()
        output = model(x, u)
        loss = loss_function(torch.max(output, dim=0)[0], gt)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        # TODO(diwei): use other progressbar
        print('step :', round((i / (len(data_loader) + + sys.float_info.epsilon)) * 100, 2),
              '% , loss :', loss.item())

    return epoch_loss / len(data_loader)


def validate(model, data_loader, loss_function, device='cpu'):
    """The interface to validate the model using validation data.

    Parameters
    ----------
    model : nn.Module
        The model to be validated.
    data_loader : DataLoader
        The interface for accessing training data and its corresponding labels 
        as the ground truth.
    loss_function : Loss nn.CrossEntropyLoss
        The loss function as a criterion for model validation.
    device : str, optional
        The computing device for validating the model (default: 'cpu').

    Returns
    -------
    float
        The loss averaged over all training samples in an epoch.
    """
    model.eval()
    epoch_loss = 0
    model.to(device)
    with torch.no_grad():
        for _, batch in enumerate(data_loader):
            x, u, gt = preprocess(batch)
            output = model(x, u)

            loss = loss_function(torch.max(output, dim=0)[0], gt)
            epoch_loss += loss.item()

    return epoch_loss / len(data_loader)


def run(total_epoch: int, best_loss: float):
    """The interface to run the training process.

    Parameters
    ----------
    total_epoch : int
        The number of iterations.
    best_loss : float
        The minimum bias between the ground truth and the predicted value.
    """
    train_losses, validation_losses = [], []
    for step in range(total_epoch):
        start_time = time.time()
        train_loss = train(model, train_data_loader, optimizer, loss)
        validation_loss = validate(model, validation_data_loader, loss)
        end_time = time.time()

        if step > top_config["train_params"]["warmup"]:
            scheduler.step(validation_loss)

        train_losses.append(train_loss)
        validation_losses.append(validation_loss)
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if validation_loss < best_loss:
            best_loss = validation_loss
            os.makedirs(top_config["train_params"]["save_path"], exist_ok=True)
            torch.save(model.state_dict(),
                       'saved/top_eleven_{0}.pt'.format(validation_loss))

        print(f'Epoch: {step + 1} | Time: {epoch_mins}m {epoch_secs}s')
        print(
            f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(
            f'\tVal Loss: {validation_loss:.3f} |  Val PPL: {math.exp(validation_loss):7.3f}')


if __name__ == '__main__':
    run(total_epoch=top_config["train_params"]
        ["epoch"], best_loss=float('inf'))
