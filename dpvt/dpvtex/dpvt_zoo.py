from neural_network import models
from neural_network.wrapper import Wrap
from dpvtex.dpvt_data import train_val_data_of_nicknames


def create_model(model_name, learning_rate):
    if model_name == "traverseNN":
        model = models.TraverseNN(learning_rate)
    elif model_name == "EncoderTraversal":
        model = models.EncoderTraversal(learning_rate)
    return model


def trained_model_str(model_name, data_name):
    return f"{model_name}-{data_name}"


def trained_model_path(model_name, data_name):
    return f"trained_models/{trained_model_str(model_name, data_name)}"


def train_model(model_name, data_name, final_checkpoint, **wrap_kwargs):
    # hyperparameter
    learning_rate = 0.01
    # model parameters
    default_params = {"batch_size": 1024, "epochs": 200}
    # Update default parameters with any provided keyword arguments
    wrap_params = {**default_params, **wrap_kwargs}
    train_data, val_data = train_val_data_of_nicknames(data_name)
    model = create_model(model_name, learning_rate)
    model_str = trained_model_str(model_name, data_name)
    wrap = Wrap(train_data, val_data, model, model_str, **wrap_params)
    wrap.train(final_checkpoint)
    return model
