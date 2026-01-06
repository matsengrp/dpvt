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
from datetime import datetime
import numpy as np

todays_date = datetime.now().strftime("%Y-%m-%d")

import optuna
import json
import os
import time

# DNA states
STATES = ["A", "G", "C", "T"]
STATE_TO_IDX = {"A": 0, "G": 1, "C": 2, "T": 3}

# Fast lookup table for DNA base -> index conversion
# Uses ASCII codes: A=65, G=71, C=67, T=84
_SEQ_LOOKUP = np.zeros(256, dtype=np.int64)
_SEQ_LOOKUP[ord('A')] = 0
_SEQ_LOOKUP[ord('G')] = 1
_SEQ_LOOKUP[ord('C')] = 2
_SEQ_LOOKUP[ord('T')] = 3


def _format_timing_report(timings, title="TraversalDataset Preprocessing Profiler Report"):
    """Format a profiler-style timing report as a list of lines.

    Args:
        timings: Dictionary of timing measurements
        title: Title for the report

    Returns:
        List of formatted lines
    """
    total = sum(v for k, v in timings.items() if not k.startswith('  '))

    lines = []
    lines.append("-" * 80)
    lines.append(title)
    lines.append("-" * 80)
    lines.append(f"{'Action':<45} {'Total time (s)':>15} {'% of total':>15}")
    lines.append("-" * 80)

    for key in ['get_tensor_representation', '  compute_dimensions',
                '  allocate_tensors', '  process_trees',
                'pad_labels', 'mask_pendant_edges']:
        if key in timings:
            t = timings[key]
            pct = 100 * t / total if total > 0 else 0
            lines.append(f"{key:<45} {t:>15.3f} {pct:>14.1f}%")

    lines.append("-" * 80)
    lines.append(f"{'Total':<45} {total:>15.3f}")
    lines.append("-" * 80)

    return lines


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
            # mask leaves, root (which is leaf) and root (which contains data
            # for edge leading to root leaf)
            mask_list = [
                not (node.is_leaf() or node.is_root() or node.up.is_root())
                for node in tree.traverse("preorder")
            ]
            masks.append(mask_list)
        mask_tensor = self.add_padding(masks, dtype=torch.bool, padding_value=False)
        return mask_tensor

    def add_padding(self, list, dtype=None, padding_value=0):
        # add padding - can be used for labels and masks
        # dtype: if specified, the tensor will be created with this dtype
        # padding_value: value to use for padding (default: 0)
        max_length = max(len(item) for item in list)
        padded_lists = [item + [padding_value] * (max_length - len(item)) for item in list]
        if dtype is not None:
            list_tensor = torch.tensor(padded_lists, dtype=dtype, device='cpu')
        else:
            list_tensor = torch.tensor(padded_lists, device='cpu')
        return list_tensor


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

    def __init__(self, trees, labels, device, profiler_dir=None):
        timings = {}

        t0 = time.perf_counter()
        self.traversal, self.mutations, tensor_timings = self.get_tensor_representation(trees)
        timings['get_tensor_representation'] = time.perf_counter() - t0
        timings.update(tensor_timings)

        t0 = time.perf_counter()
        self.labels = self.pad_labels(labels)
        timings['pad_labels'] = time.perf_counter() - t0

        t0 = time.perf_counter()
        self.mask = self.mask_pendant_edges(trees)
        timings['mask_pendant_edges'] = time.perf_counter() - t0

        self.device = device
        self.preprocessing_timings = timings
        self._print_timing_report(timings, profiler_dir)

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
        """Original implementation with per-site Python loop."""
        timings = {}

        print(f"Preprocessing {len(trees)} trees into tensor representation...")
        print("  Step 1/4: Computing maximum dimensions...")
        t0 = time.perf_counter()
        max_n_sites = max([len(tree.sequence) for tree in trees])
        max_n_nodes = max([len(list(tree.traverse())) for tree in trees])
        max_n_int_nodes = max([len(tree) - 2 for tree in trees])
        timings['  compute_dimensions'] = time.perf_counter() - t0
        print(f"    Max sites: {max_n_sites}, Max nodes: {max_n_nodes}, Max internal nodes: {max_n_int_nodes}")

        print("  Step 2/4: Allocating tensors...")
        t0 = time.perf_counter()
        mutations = torch.full(
            (len(trees), max_n_nodes, max_n_sites, 4), -1, dtype=torch.float32, device='cpu'
        )
        # from actual 0 entries representing no mutation
        traversal = torch.full(
            (len(trees), 2, max_n_int_nodes, 3), -1, dtype=torch.float32, device='cpu'
        )
        timings['  allocate_tensors'] = time.perf_counter() - t0
        tensor_size_gb = (mutations.numel() + traversal.numel()) * 4 / (1024**3)
        print(f"    Allocated {tensor_size_gb:.6f} GB of tensors")

        print("  Step 3/4: Processing trees (this may take a while)...")
        t0 = time.perf_counter()
        tree_index = 0
        report_interval = max(1, len(trees) // 20)  # Report every 5% progress
        # child and parent index in traversal
        for tree in trees:
            if tree_index % report_interval == 0:
                print(f"    Progress: {tree_index}/{len(trees)} trees ({100*tree_index/len(trees):.1f}%)")
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
        timings['  process_trees'] = time.perf_counter() - t0
        print(f"    Progress: {len(trees)}/{len(trees)} trees (100.0%)")
        print("  Step 4/4: Tensor representation complete!")

        return traversal, mutations, timings

    def get_tensor_representation_vectorized(self, trees):
        """Vectorized implementation that eliminates per-site Python loop.

        This is an optimized version of get_tensor_representation() that uses
        numpy vectorization to process all sites at once per node, rather than
        iterating through sites one-by-one. It also only touches mutation sites
        (where parent and child differ) rather than processing all sites.

        For validation, compare outputs:
            original = dataset.get_tensor_representation(trees)
            vectorized = dataset.get_tensor_representation_vectorized(trees)
            assert torch.equal(original[0], vectorized[0])  # traversal
            assert torch.equal(original[1], vectorized[1])  # mutations
        """
        timings = {}

        # Validate sequences contain only valid bases
        valid_bases = set('AGCT')
        for tree in trees:
            for node in tree.traverse():
                invalid = set(node.sequence) - valid_bases
                if invalid:
                    raise ValueError(f"Each node sequence must be in {STATES}, found: {invalid}")

        print(f"Preprocessing {len(trees)} trees into tensor representation (vectorized)...")
        print("  Step 1/4: Computing maximum dimensions...")
        t0 = time.perf_counter()
        max_n_sites = max([len(tree.sequence) for tree in trees])
        max_n_nodes = max([len(list(tree.traverse())) for tree in trees])
        max_n_int_nodes = max([len(tree) - 2 for tree in trees])
        timings['  compute_dimensions'] = time.perf_counter() - t0
        print(f"    Max sites: {max_n_sites}, Max nodes: {max_n_nodes}, Max internal nodes: {max_n_int_nodes}")

        print("  Step 2/4: Allocating tensors...")
        t0 = time.perf_counter()
        mutations = torch.full(
            (len(trees), max_n_nodes, max_n_sites, 4), -1, dtype=torch.float32, device='cpu'
        )
        # from actual 0 entries representing no mutation
        traversal = torch.full(
            (len(trees), 2, max_n_int_nodes, 3), -1, dtype=torch.float32, device='cpu'
        )
        timings['  allocate_tensors'] = time.perf_counter() - t0
        tensor_size_gb = (mutations.numel() + traversal.numel()) * 4 / (1024**3)
        print(f"    Allocated {tensor_size_gb:.6f} GB of tensors")

        print("  Step 3/4: Processing trees (vectorized)...")
        t0 = time.perf_counter()
        tree_index = 0
        report_interval = max(1, len(trees) // 20)  # Report every 5% progress
        # child and parent index in traversal
        for tree in trees:
            if tree_index % report_interval == 0:
                print(f"    Progress: {tree_index}/{len(trees)} trees ({100*tree_index/len(trees):.1f}%)")
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

                # Vectorized mutation encoding (replaces per-site loop)
                node_idx = node_index_dict[node]

                # Initialize all sites to 0 at once
                mutations[tree_index, node_idx, :n_sites, :] = 0.0

                if node.up is not None:  # non-root node
                    # Convert sequences to index arrays using numpy lookup (no Python loop)
                    node_bytes = np.frombuffer(node.sequence.encode('ascii'), dtype=np.uint8)
                    parent_bytes = np.frombuffer(node.up.sequence.encode('ascii'), dtype=np.uint8)
                    node_seq_idx = _SEQ_LOOKUP[node_bytes]
                    parent_seq_idx = _SEQ_LOOKUP[parent_bytes]

                    # Find mutation sites (where sequences differ)
                    diff_mask = node_seq_idx != parent_seq_idx

                    if np.any(diff_mask):
                        mut_sites = np.where(diff_mask)[0]
                        # Set +1 for mutation TO (node's base)
                        mutations[tree_index, node_idx, mut_sites, node_seq_idx[diff_mask]] = 1.0
                        # Set -1 for mutation FROM (parent's base)
                        mutations[tree_index, node_idx, mut_sites, parent_seq_idx[diff_mask]] = -1.0
            tree_index += 1
        timings['  process_trees'] = time.perf_counter() - t0
        print(f"    Progress: {len(trees)}/{len(trees)} trees (100.0%)")
        print("  Step 4/4: Tensor representation complete!")

        return traversal, mutations, timings

    def mask_pendant_edges(self, trees):
        # Create list of tensors, each containing the mask for one tree
        print(f"Creating pendant edge masks for {len(trees)} trees...")
        masks = [
            torch.tensor(
                [
                    not (node.is_leaf() or node.is_root() or node.up.is_root())
                    for node in tree.traverse("preorder")
                ],
                dtype=torch.bool,
                device='cpu',
            )
            for tree in trees
        ]
        # Pad and stack all masks into a single tensor
        print("  Padding and stacking masks...")
        result = pad_sequence(masks, batch_first=True, padding_value=False)
        print("  Masking complete!")
        return result

    def pad_labels(self, labels):
        print(f"Padding labels for {len(labels)} trees...")
        label_tensors = [torch.tensor(label, dtype=torch.float32, device='cpu') for label in labels]
        result = pad_sequence(label_tensors, batch_first=True, padding_value=0)
        print("  Label padding complete!")
        return result

    def _print_timing_report(self, timings, profiler_dir=None):
        """Print and optionally save a profiler-style timing report for dataset preprocessing.

        Args:
            timings: Dictionary of timing measurements
            profiler_dir: If provided, writes the report to this directory
        """
        lines = _format_timing_report(timings)
        print("\n" + "\n".join(lines) + "\n")

        if profiler_dir:
            os.makedirs(profiler_dir, exist_ok=True)
            filepath = os.path.join(profiler_dir, "preprocessing_profile.txt")
            with open(filepath, 'w') as f:
                f.write("\n".join(lines) + "\n")
            print(f"Timing report written to: {filepath}")


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
        profiling: Boolean indicating whether to use AdvancedProfiler for
            CPU profiling
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
        # Use fewer workers to avoid GPU memory issues with large datasets
        # pin_memory helps transfer data to GPU efficiently
        num_workers = 2 if self.device in ["cuda", "gpu"] else 10
        self.train_loader = DataLoader(
            train_data,
            batch_size=self.batch_size,
            collate_fn=custom_collate,
            num_workers=num_workers,
            drop_last=droplast,
            pin_memory=True if self.device in ["cuda", "gpu"] else False,
        )
        self.val_loader = DataLoader(
            val_data,
            batch_size=self.batch_size,
            collate_fn=custom_collate,
            num_workers=num_workers,
            drop_last=droplast,
            pin_memory=True if self.device in ["cuda", "gpu"] else False,
        )
        self.test_loader = DataLoader(
            test_data,
            batch_size=self.batch_size,
            collate_fn=custom_collate,
            num_workers=num_workers,
            drop_last=droplast,
            pin_memory=True if self.device in ["cuda", "gpu"] else False,
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

        # Build callbacks list
        callbacks = [checkpoint_callback, early_stop_callback] + added_callbacks

        # Set up profiler
        profiler = None
        if self.profiling:
            # Extract just the basename for the filename, as AdvancedProfiler
            # combines dirpath + filename and doesn't handle absolute paths well
            profiler_filename = os.path.basename(self.log_path)
            profiler_dir = "profiler_output/" + self.device
            profiler = AdvancedProfiler(
                dirpath=profiler_dir,
                filename=profiler_filename,
            )
            # Write preprocessing timings from datasets to profiler directory
            self._write_preprocessing_timings(
                profiler_dir, train_data, val_data, test_data, profiler_filename
            )

        self.trainer = L.Trainer(
            accelerator=self.device,
            devices=1,
            logger=logger,
            log_every_n_steps=1,
            max_epochs=self.epochs,
            # limit_train_batches=1,
            callbacks=callbacks,
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

    def _write_preprocessing_timings(
        self, profiler_dir, train_data, val_data, test_data, profiling_filename
    ):
        """Write preprocessing timing reports from datasets to the profiler directory.

        Args:
            profiler_dir: Directory to write profiler output files.
            train_data: Training dataset.
            val_data: Validation dataset.
            test_data: Test dataset.
            profiling_filename: Base filename (e.g., 'TraverseNN-benchmark_train-Param0')
                to include in output filenames for identification.
        """
        os.makedirs(profiler_dir, exist_ok=True)

        datasets = [
            ('train', train_data),
            ('val', val_data),
            ('test', test_data),
        ]

        for name, dataset in datasets:
            if hasattr(dataset, 'preprocessing_timings'):
                title = f"TraversalDataset Preprocessing Profiler Report ({name})"
                lines = _format_timing_report(dataset.preprocessing_timings, title)
                filepath = os.path.join(
                    profiler_dir, f"preprocessing_{name}-{profiling_filename}.txt"
                )
                with open(filepath, 'w') as f:
                    f.write("\n".join(lines) + "\n")
                print(f"Preprocessing timing report written to: {filepath}")


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

        # Set up profiler
        profiler = None
        if self.profiling:
            # Extract just the basename for the filename, as AdvancedProfiler
            # combines dirpath + filename and doesn't handle absolute paths well
            profiler_filename = os.path.basename(self.log_path)
            profiler_dir = "profiler_output/" + self.device
            profiler = AdvancedProfiler(
                dirpath=profiler_dir,
                filename=profiler_filename,
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
