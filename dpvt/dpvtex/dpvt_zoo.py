from neural_network import models
from neural_network.wrapper import Wrap, HyperWrap
from dpvtex.dpvt_data import train_val_data_of_nicknames


def create_model(model_name):
    if model_name == "traverseNN":
        model = models.TraverseNN
    elif model_name == "TransformerEncoderTraversal":
        model = models.TransformerEncoderTraversal
    return model


def trained_model_str(model_name, data_name):
    return f"{model_name}-{data_name}"


def trained_model_path(model_name, data_name):
    return f"trained_models/{trained_model_str(model_name, data_name)}"


def best_model_params_path(model_name, data_name):
    return f"hyper_checkpoints/{trained_model_str(model_name, data_name)}"


def train_model(
    model_name, data_name, final_checkpoint, test_checkpoint, **wrap_kwargs
):
    # hyperparameters (only used if no hyperparameter testing done)
    default_params = {
        "learning_rate": 0.01,
        "batch_size": 1024,
        "epochs": 200,
    }
    # Update default parameters with any provided keyword arguments
    wrap_params = {**default_params, **wrap_kwargs}
    train_data, val_data, test_data = train_val_data_of_nicknames(data_name)
    model = create_model(model_name)
    model_str = trained_model_str(model_name, data_name)
    wrap = Wrap(train_data, val_data, test_data, model, model_str, **wrap_params)
    wrap.train(final_checkpoint)
    wrap.test(test_checkpoint)
    return model


def optimize_hyperparameters(model_name, data_name, best_model_hparams_filepath):
    train_data, val_data, _ = train_val_data_of_nicknames(data_name)
    model = create_model(model_name)
    model_str = trained_model_str(model_name, data_name)
    hyper_wrap = HyperWrap(
        model, train_data, val_data, model_str, n_trials=10
    )  # n_trials chosen small for testing
    hyper_wrap.optuna_optimize(best_model_hparams_filepath)
    return model
