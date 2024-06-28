import os
import sys

import torch.nn as nn

# TODO: add a flag to opt for different methods for parameter initialization
# TODO: add unittests
def initialize_model_parameters(model):
    for param in model.parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)
    return model

# TODO: deep dive
# TODO: add unittests
def check_model_file_path(model_file_path):
    if not os.path.exists(model_file_path):
        file_dir, file_name = os.path.split(model_file_path)
        file_dir = os.path.abspath(file_dir)
        print("Checking model file directory: '%s'." % file_dir)
        if not os.path.exists(file_dir):
            print("The directory '%s' does not exist. Try to create a new one." % file_dir)
            try:
                os.makedirs(file_dir, exist_ok = True)
                print("Directory '%s' created successfully." % file_dir)
            except OSError as error:
                print("Directory '%s' cannot be created." % file_dir)
        else:
            print("The directory '%s' already exists." % file_dir)
        print("Checking model file name: '%s'." % file_name)
        file_name, file_ext = os.path.splitext(file_name)
        if file_ext == '' or file_ext not in ['.pt', '.pth']:
            sys.exit('Invalid model file extension.')
        model_file_path = os.path.join(file_dir, file_name, file_ext)
        print("The model file '%s' does not exist, but is valid to be created." % model_file_path)
    else:
        print("The model file '%s' already exists. Possible to be overwritten." % model_file_path)
