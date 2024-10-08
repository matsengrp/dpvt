import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import lightning as L
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import AdvancedProfiler
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import Dataset
from ete3 import Tree
from datetime import datetime
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
        items is a list of (input, output, mask) tuples, where `input` is an ete3.Tree,
        `output` is a float, and `mask` is a boolean
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
    def __init__(self, data, labels):
        self.data = data
        self.labels = self.add_padding(labels).to(torch.float64)
        self.mask = self.mask_pendant_edges(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.mask[idx]

    def mask_pendant_edges(self, trees):
        masks = []
        for tree in trees:
            # mask leaves, root (which is leaf) and root (which contains data for edge
            # leading to root leaf)
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


class TraversalDataset(Dataset):
    """Dataset where trees and mutations on trees are encoded as tensors.
    The tensor `traversal` for contains for each tree all triples
    (child1, child2, int_node) and all triples (sibling, parent, int_node)
    for every internal node int_node that is below an internal edge.
    Mutations are encoded by tensors of length 4 for each site with
    entries 0,1, and -1 where -1 indicated the base that has been lost
    from parent and 1 the base it mutated to. The bases are ordered
    A,G,C,T. E.g. (1,0,-1,0) indicates mutation C -> A.
    Labels indicate edges that are MP (0) vs non-MP (1) and mask
    contains boolean values indicating whether edges are adjacent to
    leaves"""

    def __init__(self, trees, labels, device):
        self.traversal, self.mutations = self.get_tensor_representation(trees)
        self.labels = self.pad_labels(labels)
        self.mask = self.mask_pendant_edges(trees)
        self.device = device

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
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
        mutations = torch.empty(len(trees), max_n_nodes, max_n_sites, 4)
        mutations.fill_(-1)  # pad with -1 so we can distinguish padding
        # from actual 0 entries representing no mutation
        traversal = torch.empty(len(trees), 2, max_n_int_nodes, 3)
        traversal.fill_(-1)
        tree_index = 0
        # child and parent index in traversal
        for tree in trees:
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
        return traversal, mutations

    def mask_pendant_edges(self, trees):
        max_n_nodes = max([len(list(tree.traverse())) for tree in trees])
        masks = torch.full((len(trees), max_n_nodes), False, dtype=torch.bool)
        i = 0
        for tree in trees:
            # mask leaves, root (which is leaf) and root (which contains data for edge
            # leading to root leaf)
            mask_list = [
                not (node.is_leaf() or node.is_root() or node.up.is_root())
                for node in tree.traverse("preorder")
            ]
            remaining_spots = max_n_nodes - len(mask_list)
            mask_list += [False for i in range(remaining_spots)]
            masks[i] = torch.tensor(mask_list)
            i += 1
        return masks

    def pad_labels(self, labels):
        max_length = max([len(label) for label in labels])
        padded_labels = torch.zeros(len(labels), max_length)
        i = 0
        for label in labels:
            to_fill = max_length - len(label)
            label += [0 for i in range(to_fill)]
            padded_labels[i] = torch.tensor(label)
            i += 1
        return padded_labels


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
        device="cpu",
        batch_size=1024,
        learning_rate=0.005,
        feature_length=32,
        dim_mlp_layers=32,
        epochs=200,
        hyperparameter_path="",
        profiling=False,
        accum_grad_batches=1,
    ):
        self.log_path = log_path
        self.epochs = epochs
        if device == "cpu-tree-dataset":
            self.device = "cpu"
        else:
            self.device = device
        self.profiling = profiling
        self.accum_grad_batches = accum_grad_batches

        # If hyperparameter tuning has been done, read hyperparameters and use them from
        # training
        if hyperparameter_path:
            print("Using best hyperparameters for ", log_path)
            with open(hyperparameter_path) as f:
                best_hyperparams = json.load(f)
            self.batch_size = best_hyperparams["batch_size"]
            self.learning_rate = best_hyperparams["learning_rate"]
            self.feature_length = best_hyperparams["feature_length"]
            self.dim_mlp_layers = best_hyperparams["dim_mlp_layers"]
            self.accum_grad_batches = best_hyperparams["accum_grad_batches"]
        else:
            print("Use default parameters for ", log_path)
            # Initialize model with specified parameters
            self.batch_size = batch_size
            self.learning_rate = learning_rate
            self.feature_length = feature_length
            self.dim_mlp_layers = dim_mlp_layers
        if isinstance(model, type):
            # `model` is a class
            self.model = model(self.learning_rate)
        else:
            # `model` is an instance of a class
            self.model = model

        droplast=False
        if self.batch_size <= 64:
            # drop last batch if we observe small batch size
            droplast = True
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

        logger = TensorBoardLogger("lightning_logs/" + self.device + "_" + str(todays_date), name=self.log_path)
        checkpoint_callback = ModelCheckpoint(every_n_epochs=1, save_top_k=1)
        # early stopping if overfitting occurs
        early_stop_callback = EarlyStopping(
            monitor="val_loss",
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
            log_every_n_steps=1,
            max_epochs=self.epochs,
            callbacks=[checkpoint_callback, early_stop_callback],
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
    A class for hyperparameter optimization.
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
        accum_grad_batches = trial.suggest_categorical("accum_grad_batches", range(1, 10))
        # epochs = trial.suggest_categorical("epochs", range(1,300))
        epochs = 200
        feature_length = trial.suggest_categorical("feature_length", [2**x for x in range(2,10)])
        dim_mlp_layers = trial.suggest_categorical("dim_mlp_layers", [2**x for x in range(2,10)])

        # Setup model, data, and trainer
        model = self.model(learning_rate, feature_length, dim_mlp_layers)
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
            callbacks=[checkpoint_callback, early_stop_callback],
            profiler=profiler,
            accumulate_grad_batches=accum_grad_batches
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
        Function to perform hyperparameter optimization
        Args:
            hyperparams_filename: json file in which to store best hyperparameters
        """
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, self.n_trials)

        best_hyperparameters = study.best_trial.params
        print(best_hyperparameters)
        with open(hyperparams_filename, "w") as f:
            json.dump(best_hyperparameters, f)

        print("Number of finished trials:", len(study.trials))
        print("Best trial:", study.best_trial.params)
