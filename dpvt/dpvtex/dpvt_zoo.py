import lightning as L
import torch
import sys

sys.path.append("..")

from neural_network import models
from neural_network.wrappers import Wrap
from dpvtex.dpvt_data import train_val_data_of_nicknames

epochs=200


def create_model(model_name):
    if model_name == "traverseNN":
        model = models.TraverseNN()
    return model


def trained_model_str(model_name, data_name):
    return f"{model_name}-{data_name}"


def trained_model_path(model_name, data_name):
    return f"trained_models/{trained_model_str(model_name, data_name)}"


def custom_collate(items):
    """
    Args:
        items is a list of (input, output) pairs, where `input` is an ete3.Tree and
        `output` is a float
    """
    return [item[0] for item in items], torch.tensor([item[1] for item in items])


def train_model(model_name, data_name, final_checkpoint):
    train_data, val_data = train_val_data_of_nicknames(data_name)
    model = create_model(model_name)
    wrap = Wrap(train_data, val_data, model, "lightning_logs")
    wrap.train(epochs, final_checkpoint)
    return model
