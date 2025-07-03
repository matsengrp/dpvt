import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import lightning as L
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import AdvancedProfiler
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import Dataset
from ete3 import Tree
import numpy as np
from datetime import datetime
from tqdm import tqdm
import pickle
import os

todays_date = datetime.now().strftime("%Y-%m-%d")

import optuna
import json

# DNA states
STATES = ["A", "G", "C", "T"]
STATE_TO_IDX = {"A": 0, "G": 1, "C": 2, "T": 3}
n_states = len(STATES)


def custom_collate(items):
    """
    Args:
        items is a list of (input, output, mask) tuples, where `input` is an
        ete3.Tree, `output` is a float, and `mask` is a boolean
    """
    if type(items[0][0]) == Tree:
        return (
            [item[0] for item in items],
            torch.stack([item[1] for item in items]),
            torch.stack([item[2] for item in items]),
        )
    else:
        return (
            torch.stack([item[0] for item in items]),
            torch.stack([item[1] for item in items]),
            torch.stack([item[2] for item in items]),
            torch.stack([item[3] for item in items]),
        )


class TreeDataset(Dataset):
    def __init__(self, data, labels, preprocessed_path=None):
        self.data = data
        self.labels = self.add_padding(labels).to(torch.float64)
        self.mask = self.mask_pendant_edges(data)

        # Load preprocessed data if path is provided
        if preprocessed_path:
            self.load_preprocessed_data(preprocessed_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.mask[idx]

    def mask_pendant_edges(self, trees):
        masks = []
        for tree in trees:
            # mask leaves, root (which is leaf) and root (which contains data
            # for edge leading to root leaf)
            mask_list = [
                not (node.is_leaf() or node.is_root() or node.up.is_root())
                for node in tree.traverse("preorder")
            ]
            masks.append(mask_list)
        mask_tensor = self.add_padding(masks)
        return mask_tensor

    def add_padding(self, list):
        # add padding with zeros - can be used for labels and masks
        max_length = max(len(item) for item in list)
        padded_lists = [item + [0] * (max_length - len(item)) for item in list]
        list_tensor = torch.tensor(padded_lists)
        return list_tensor

    def save_preprocessed_data(self, file_path):
        """Save the preprocessed tensor representations to a file."""
        with open(file_path, "wb") as f:
            pickle.dump((self.data, self.labels, self.mask), f)

    def load_preprocessed_data(self, file_path):
        """Load preprocessed tensor representations from a file."""
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                self.data, self.labels, self.mask = pickle.load(f)
        else:
            print(f"File {file_path} does not exist.")


class TraversalDataset(Dataset):
    """Dataset where trees and mutations on trees are encoded as tensors.
    The tensor `traversal` for contains for each tree all triples (child1,
    child2, int_node) and all triples (sibling, parent, int_node) for every
    internal node int_node that is below an internal edge. Mutations are encoded
    by tensors of length 4 for each site with entries 0,1, and -1 where -1
    indicated the base that has been lost from parent and 1 the base it mutated
    to. The bases are ordered A,G,C,T. E.g. (1,0,-1,0) indicates mutation C ->
    A. Labels indicate edges that are MP (0) vs non-MP (1) and mask contains
    boolean values indicating whether edges are adjacent to leaves"""

    def __init__(
        self,
        trees=None,
        labels=None,
        traversal=None,
        mutations=None,
        traversal_labels=None,
        mask=None,
        device="cpu",
        preprocessed_path=None,
    ):
        # Always keep data on CPU, will be moved to GPU in batches during training
        self.device = "cpu"

        # Load preprocessed data if path is provided
        if preprocessed_path:
            self.load_preprocessed_data(preprocessed_path)
        elif trees is not None and labels is not None:
            self.traversal, self.mutations = self.get_tensor_representation(trees)
            self.labels = self.pad_labels(labels)
            self.mask = self.mask_pendant_edges(trees)
        else:
            self.traversal = traversal
            self.mutations = mutations
            self.labels = traversal_labels
            self.mask = mask

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Handle memory-mapped arrays by copying to avoid reference issues
        if isinstance(self.traversal, np.ndarray):
            traversal = torch.from_numpy(self.traversal[idx].copy()).float()
            mutations = torch.from_numpy(self.mutations[idx].copy()).float()
            labels = torch.from_numpy(self.labels[idx].copy()).float()
            mask = torch.from_numpy(self.mask[idx].copy()).bool()
        else:
            # Regular tensor access
            traversal = self.traversal[idx]
            mutations = self.mutations[idx]
            labels = self.labels[idx]
            mask = self.mask[idx]
        return (
            traversal,
            mutations,
            labels,
            mask,
        )

    def get_tensor_representation(self, trees):
        max_n_sites = max([len(tree.sequence) for tree in trees])
        max_n_nodes = max([len(list(tree.traverse())) for tree in trees])
        max_n_int_nodes = max([len(tree) - 2 for tree in trees])
        mutations = torch.full(
            (len(trees), max_n_nodes, max_n_sites, 4), -1, dtype=torch.float32
        )
        # from actual 0 entries representing no mutation
        traversal = torch.full(
            (len(trees), 2, max_n_int_nodes, 3), -1, dtype=torch.float32
        )
        tree_index = 0
        # child and parent index in traversal
        print(f"Start pre-processing {len(trees)} trees")
        for tree in tqdm(trees, desc="Converting trees to tensor representation"):
            node_index_dict = {
                node: index
                for (node, index) in zip(
                    tree.traverse("preorder"), range(len(list(tree.traverse())))
                )
            }  # save index for every node (preorder) to easily find it in traversals
            node_in_tensor_index = 0
            for node in tree.traverse("postorder"):
                if not (node.is_leaf() or node.is_root() or node.up.is_root()):
                    children = node.get_children()
                    traversal[tree_index, 0, node_in_tensor_index, :] = torch.tensor(
                        [
                            node_index_dict[children[0]],
                            node_index_dict[children[1]],
                            node_index_dict[node],
                        ]
                    )
                    node_in_tensor_index += 1

            node_in_traversal_index = 0
            n_sites = len(tree.sequence)
            # downward traversal to assign mutations and traversal
            for node in tree.traverse("preorder"):
                # fill downward traversal (preorder) entries
                if not (node.is_leaf() or node.is_root() or node.up.is_root()):
                    traversal[tree_index, 1, node_in_traversal_index, :] = torch.tensor(
                        [
                            node_index_dict[node.up],
                            node_index_dict[node.get_sisters()[0]],
                            node_index_dict[node],
                        ]
                    )
                    node_in_traversal_index += 1
                for site_index in range(n_sites):
                    mutations[tree_index, node_index_dict[node], site_index, :] = 0.0
                    if node.up is None:  # node is root
                        pass
                    else:  # non-root node
                        n_seq = node.sequence[site_index]
                        p_seq = node.up.sequence[site_index]
                        try:
                            mutations[
                                tree_index,
                                node_index_dict[node],
                                site_index,
                                STATE_TO_IDX[n_seq],
                            ] += 1
                            mutations[
                                tree_index,
                                node_index_dict[node],
                                site_index,
                                STATE_TO_IDX[p_seq],
                            ] -= 1
                        except KeyError:
                            raise ValueError(f"Each node sequence must be in {STATES}")
            tree_index += 1
        print("Finished pre-processing trees")
        return traversal, mutations

    def mask_pendant_edges(self, trees):
        # Create list of tensors, each containing the mask for one tree
        masks = [
            torch.tensor(
                [
                    not (node.is_leaf() or node.is_root() or node.up.is_root())
                    for node in tree.traverse("preorder")
                ],
                dtype=torch.bool,
            )
            for tree in trees
        ]
        # Pad and stack all masks into a single tensor
        return pad_sequence(masks, batch_first=True, padding_value=False)

    def pad_labels(self, labels):
        label_tensors = [torch.tensor(label, dtype=torch.float32) for label in labels]
        return pad_sequence(label_tensors, batch_first=True, padding_value=0)

    def save_preprocessed_data(self, file_path):
        """Save the preprocessed tensor representations to a file."""
        with open(file_path, "wb") as f:
            pickle.dump((self.traversal, self.mutations, self.labels, self.mask), f)

    def load_preprocessed_data(self, file_path):
        """Load preprocessed tensor representations from a file using memory-mapped arrays for large datasets."""
        if not os.path.exists(file_path):
            print(f"File {file_path} does not exist.")
            return

        # Check if memory-mapped files exist
        memmap_files = [
            file_path.replace(".p", "_traversal.npy"),
            file_path.replace(".p", "_mutations.npy"),
            file_path.replace(".p", "_labels.npy"),
            file_path.replace(".p", "_mask.npy"),
        ]

        if all(os.path.exists(f) for f in memmap_files):
            # Load from memory-mapped files
            print(f"Loading from memory-mapped files for {file_path}")
            self.traversal = np.load(memmap_files[0], mmap_mode="r")
            self.mutations = np.load(memmap_files[1], mmap_mode="r")
            self.labels = np.load(memmap_files[2], mmap_mode="r")
            self.mask = np.load(memmap_files[3], mmap_mode="r")
        else:
            # First time: load pickle and create memory-mapped files
            print(f"Converting {file_path} to memory-mapped format...")
            with open(file_path, "rb") as f:
                self.traversal, self.mutations, self.labels, self.mask = pickle.load(f)

            # Save as memory-mapped arrays for future use
            np.save(memmap_files[0], self.traversal)
            np.save(memmap_files[1], self.mutations)
            np.save(memmap_files[2], self.labels)
            np.save(memmap_files[3], self.mask)
            print("Conversion complete - future loads will be memory-efficient")


class Wrap:
    """
    Wrapper class for tree traversal neural network models.
    args:
        train_data: Dataset nickname for training
        val_data: Dataset nickname for validation
        test_data: Dataset nickname for testing
        model: Model class or instance
        log_path: Path to save logs and checkpoints
        device: Device to use for training (e.g., "cpu", "cuda")
        batch_size: Batch size for training
        learning_rate: Learning rate for the optimizer
        feature_length: Length of the feature vector
        dim_mlp_layers: Dimension of MLP layers
        epochs: Number of epochs for training
        hyperparameter_path: Path to a JSON file with hyperparameters, which
            replace the default parameters in the input here
        profiling: Boolean indicating whether to use profiling
        accum_grad_batches: Number of batches to accumulate gradients over
        timestamp: Timestamp for logging
        added_callbacks: List of additional callbacks for the trainer
    """

    def __init__(
        self,
        train_data,
        val_data,
        test_data,
        model,
        log_path,
        device="cpu",
        batch_size=1024,
        learning_rate=0.005,
        feature_length=32,
        dim_mlp_layers=32,
        epochs=200,
        hyperparameter_path="",
        profiling=False,
        accum_grad_batches=1,
        timestamp=str(todays_date),
        added_callbacks=[],
    ):
        self.log_path = log_path
        if device == "cpu-tree-dataset":
            self.device = "cpu"
        else:
            self.device = device
        self.profiling = profiling
        self.accum_grad_batches = accum_grad_batches

        # If hyperparameter tuning has been done, read hyperparameters and use
        # them from training
        print("Hyperparameter path:", hyperparameter_path)
        if hyperparameter_path:
            print("Using best hyperparameters for ", log_path)
            with open(hyperparameter_path) as f:
                best_hyperparams = json.load(f)
            self.batch_size = best_hyperparams["batch_size"]
            self.learning_rate = best_hyperparams["learning_rate"]
            self.feature_length = best_hyperparams["feature_length"]
            self.dim_mlp_layers = best_hyperparams["dim_mlp_layers"]
            self.accum_grad_batches = best_hyperparams["accum_grad_batches"]
            self.epochs = best_hyperparams["epochs"]
            print(f"hyperparameters: {best_hyperparams}")
        else:
            print("Use default parameters for ", log_path)
            # Initialize model with specified parameters
            self.batch_size = batch_size
            self.learning_rate = learning_rate
            self.feature_length = feature_length
            self.dim_mlp_layers = dim_mlp_layers
            self.epochs = epochs
        if isinstance(model, type):
            # `model` is a class
            self.model = model(
                self.learning_rate, self.feature_length, self.dim_mlp_layers
            )
        else:
            # `model` is an instance of a class
            self.model = model

        droplast = False
        # if self.batch_size <= 64: # drop last batch if we observe small batch
        #     size droplast = True
        self.train_loader = DataLoader(
            train_data,
            batch_size=self.batch_size,
            collate_fn=custom_collate,
            num_workers=10,
            drop_last=droplast,
        )
        self.val_loader = DataLoader(
            val_data,
            batch_size=self.batch_size,
            collate_fn=custom_collate,
            num_workers=10,
            drop_last=droplast,
        )
        self.test_loader = DataLoader(
            test_data,
            batch_size=self.batch_size,
            collate_fn=custom_collate,
            num_workers=10,
            drop_last=droplast,
        )

        logger = TensorBoardLogger(
            f"lightning_logs/{self.device}_{timestamp}", name=self.log_path
        )
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.log_path,
            filename="{epoch}-{val_loss:.2f}",
            every_n_epochs=10,
            save_top_k=-1,
        )
        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            patience=5,
            mode="min",
        )
        profiler = None
        if self.profiling:
            profiler = AdvancedProfiler(
                dirpath="profiler_output/" + self.device, filename=self.log_path
            )
        self.trainer = L.Trainer(
            accelerator=self.device,
            devices=1,
            logger=logger,
            log_every_n_steps=1,
            max_epochs=self.epochs,
            # limit_train_batches=1,
            callbacks=[checkpoint_callback, early_stop_callback] + added_callbacks,
            profiler=profiler,
            accumulate_grad_batches=self.accum_grad_batches,
        )

    def train(self, checkpoint):
        # train and save trained model
        self.trainer.fit(self.model, self.train_loader, self.val_loader)
        self.trainer.save_checkpoint(checkpoint)

    def test(self, trained_model_ckpt, checkpoint):
        # test and save model
        result = self.trainer.test(self.model, self.test_loader, trained_model_ckpt)
        self.trainer.save_checkpoint(checkpoint)
        return result


class HyperWrap:
    """
    Wrapper class for hyperparameter optimization of tree traversal neural network models.
    args:
        train_data: Dataset nickname for training
        val_data: Dataset nickname for validation
        model: Model class or instance
        log_path: Path to save logs and checkpoints
        device: Device to use for training (e.g., "cpu", "cuda")
        n_trials: Number of trials for hyperparameter optimization
        checkpoint_dir: Directory to save checkpoints
        profiling: Boolean indicating whether to use profiling
        added_callbacks: List of additional callbacks for the trainer
    """

    def __init__(
        self,
        model,
        train_data,
        val_data,
        log_path,
        device="cpu",
        n_trials=10,
        checkpoint_dir="hyper_checkpoints/",
        profiling=False,
        added_callbacks=[],
    ):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.log_path = log_path
        if device == "cpu-tree-dataset":
            self.device = "cpu"
        else:
            self.device = device
        self.n_trials = n_trials
        self.checkpoint_dir = checkpoint_dir
        self.profiling = profiling
        self.added_callbacks = added_callbacks

    def objective(self, trial):
        """
        Objective function for Optuna Hyperparameter Optimization. Returns
        validation loss.
        """
        # Define hyperparameter search space
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical(
            "batch_size", [2**x for x in range(1, 6)]
        )
        accum_grad_batches = trial.suggest_categorical(
            "accum_grad_batches", range(1, 10)
        )
        epochs = trial.suggest_categorical("epochs", range(1, 300))
        # epochs = 200
        feature_length = trial.suggest_categorical(
            "feature_length", [2**x for x in range(2, 10)]
        )
        dim_mlp_layers = trial.suggest_categorical(
            "dim_mlp_layers", [2**x for x in range(2, 10)]
        )

        # Setup model, data, and trainer
        model = self.model(learning_rate, feature_length, dim_mlp_layers)
        train_loader = DataLoader(
            self.train_data, batch_size=batch_size, collate_fn=custom_collate
        )
        val_loader = DataLoader(
            self.val_data, batch_size=batch_size, collate_fn=custom_collate
        )
        logger = TensorBoardLogger(self.checkpoint_dir, name=self.log_path)
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.log_path,
            filename="{epoch}-{val_loss:.2f}",
            every_n_epochs=10,
            save_top_k=1,
        )
        early_stop_callback = EarlyStopping(
            monitor="val_loss",  # Metric to monitor
            patience=3,  # Number of epochs with no improvement after which training will be stopped
            mode="min",  # Stop training when the quantity monitored has stopped decreasing
        )
        profiler = None
        if self.profiling:
            profiler = AdvancedProfiler(
                dirpath="profiler_output/" + self.device, filename=self.log_path
            )
        self.trainer = L.Trainer(
            accelerator=self.device,
            devices=1,
            logger=logger,
            max_epochs=epochs,
            # limit_train_batches=1,
            callbacks=[checkpoint_callback, early_stop_callback] + self.added_callbacks,
            profiler=profiler,
            accumulate_grad_batches=accum_grad_batches,
        )

        # Train the model
        self.trainer.fit(model, train_loader, val_loader)

        # Return the metric to optimize
        return self.trainer.callback_metrics["val_loss"].item()

    def optuna_optimize(
        self,
        hyperparams_filename,
    ):
        """
        Function to perform hyperparameter optimization Args:
            hyperparams_filename: json file in which to store best
            hyperparameters
        """
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, self.n_trials, gc_after_trial=True, n_jobs=4)

        best_hyperparameters = study.best_trial.params
        with open(hyperparams_filename, "w") as f:
            json.dump(best_hyperparameters, f)

        print("Number of finished trials:", len(study.trials))
        print("Best trial:", study.best_trial.params)


class Wraplet:
    """
    Lightweight wrapper class for baseline models that don't need to be trained.
    args:
        test_data: Dataset for testing
        model: Model class or instance
    """

    def __init__(self, test_data, model, device="cpu"):
        self.device = device
        self.batch_size = len(test_data)

        # Initialize the model
        if isinstance(model, type):
            # `model` is a class
            self.model = model()
        else:
            # `model` is an instance of a class
            self.model = model

        # Create test loader
        self.test_loader = DataLoader(
            test_data,
            batch_size=self.batch_size,
            collate_fn=custom_collate,
            num_workers=10,
            drop_last=False,  # We typically want all data for evaluation
        )

        # Set up a simple trainer for testing only
        self.trainer = L.Trainer(
            accelerator=self.device,
            devices=1,
            enable_checkpointing=False,  # No need to save checkpoints for baseline model
            logger=False,  # No need for logging during testing
        )

    def test(self):
        """Test the baseline model and return results"""
        results = self.trainer.test(self.model, self.test_loader)
        return results
