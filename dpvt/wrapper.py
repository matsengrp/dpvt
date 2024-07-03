import torch
from torch.utils.data import DataLoader
import lightning as L
from pytorch_lightning.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

import optuna
import json


def custom_collate(items):
    """
    Args:
        items is a list of (input, output, mask) tuples, where `input` is an ete3.Tree,
        `output` is a float, and `mask` is a boolean
    """
    return (
        [item[0] for item in items],
        torch.tensor([item[1] for item in items]),
        torch.tensor([item[2] for item in items]),
    )


class Wrap:
    """
    A class for wrapping a neural network model and a dataset.
    """

    def __init__(
        self,
        train_data,
        val_data,
        test_data,
        model,
        log_path,
        batch_size=1024,
        learning_rate=0.005,
        epochs=200,
        hyperparameter_path="",
    ):
        self.log_path = log_path
        self.epochs = epochs

        # If hyperparameter tuning has been done, read hyperparameters and use them from
        # training
        if hyperparameter_path:
            print("Using best hyperparameters for ", log_path)
            with open(hyperparameter_path) as f:
                best_hyperparams = json.load(f)
            self.batch_size = best_hyperparams["batch_size"]
            self.learning_rate = best_hyperparams["learning_rate"]
        else:
            print("Use default parameters for ", log_path)
            # Initialize model with specified parameters
            self.batch_size = batch_size
            self.learning_rate = learning_rate
        if isinstance(model, type):
            # `model` is a class
            self.model = model(self.learning_rate)
        else:
            # `model` is an instance of a class
            self.model = model

        self.train_loader = DataLoader(
            train_data, batch_size=self.batch_size, collate_fn=custom_collate
        )
        self.val_loader = DataLoader(
            val_data, batch_size=self.batch_size, collate_fn=custom_collate
        )
        self.test_loader = DataLoader(
            test_data, batch_size=self.batch_size, collate_fn=custom_collate
        )

        logger = TensorBoardLogger("lightning_logs", name=self.log_path)
        checkpoint_callback = ModelCheckpoint(every_n_epochs=10, save_top_k=-1)
        # early stopping if overfitting occurs
        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            patience=20,  # Number of epochs with no improvement after which training will be stopped
            mode="min",  # Stop training when the quantity monitored has stopped decreasing
        )
        self.trainer = L.Trainer(
            logger=logger,
            max_epochs=self.epochs,
            log_every_n_steps=1,
            callbacks=[checkpoint_callback, early_stop_callback],
        )

    def train(self, checkpoint):
        # train and save trained model
        self.trainer.fit(self.model, self.train_loader, self.val_loader)
        self.trainer.save_checkpoint(checkpoint)

    def test(self, checkpoint):
        # test and save model
        self.model.eval()
        result = self.trainer.test(self.model, self.test_loader)
        self.trainer.save_checkpoint(checkpoint)
        return result


class HyperWrap:
    """
    A class for hyperparameter optimization.
    """

    def __init__(
        self,
        model,
        train_data,
        val_data,
        log_path,
        epochs=200,
        n_trials=10,
        checkpoint_dir="hyper_checkpoints/",
    ):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.log_path = log_path
        self.epochs = epochs
        self.n_trials = n_trials
        self.checkpoint_dir = checkpoint_dir

    def objective(self, trial):
        """
        Objective function for Optuna Hyperparameter Optimization.
        Returns validation loss.
        """
        # Define hyperparameter search space
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical(
            "batch_size", [2**x for x in range(4, 10)]
        )

        # Setup model, data, and trainer
        model = self.model(learning_rate)
        train_loader = DataLoader(
            self.train_data, batch_size=batch_size, collate_fn=custom_collate
        )
        val_loader = DataLoader(
            self.val_data, batch_size=batch_size, collate_fn=custom_collate
        )
        logger = TensorBoardLogger(self.checkpoint_dir, name=self.log_path)
        checkpoint_callback = ModelCheckpoint(every_n_epochs=10, save_top_k=-1)
        early_stop_callback = EarlyStopping(
            monitor="val_loss",  # Metric to monitor
            patience=20,  # Number of epochs with no improvement after which training will be stopped
            mode="min",  # Stop training when the quantity monitored has stopped decreasing
        )
        trainer = L.Trainer(
            logger=logger,
            max_epochs=self.epochs,
            callbacks=[checkpoint_callback, early_stop_callback],
        )

        # Train the model
        trainer.fit(model, train_loader, val_loader)

        # Return the metric to optimize
        return trainer.callback_metrics["val_loss"].item()

    def optuna_optimize(
        self,
        hyperparams_filename,
    ):
        """
        Function to perform hyperparameter optimization
        Args:
            hyperparams_filename: json file in which to store best hyperparameters
        """
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, self.n_trials)

        best_hyperparameters = study.best_trial.params
        with open(hyperparams_filename, "w") as f:
            json.dump(best_hyperparameters, f)

        print("Number of finished trials:", len(study.trials))
        print("Best trial:", study.best_trial.params)
