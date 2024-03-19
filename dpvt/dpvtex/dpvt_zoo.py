import lightning as L
import torch

from neural_network import models
from neural_network.wrapper import Wrap
from dpvtex.dpvt_data import train_val_data_of_nicknames


def create_model(model_name):
    if model_name == "traverseNN":
        model = models.TraverseNN(learning_rate=0.01)
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


def train_model(model_name, data_name, final_checkpoint, **wrap_kwargs):
    default_params = {"batch_size": 16, "epochs": 100, "learning_rate": 0.01}
    # Update default parameters with any provided keyword arguments
    wrap_params = {**default_params, **wrap_kwargs}
    train_data, val_data = train_val_data_of_nicknames(data_name)
    model = create_model(model_name)
    wrap = Wrap(train_data, val_data, model, "lightning_logs", **wrap_params)
    wrap.train(final_checkpoint)
    return model
