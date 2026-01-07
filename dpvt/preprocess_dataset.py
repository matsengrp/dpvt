#!/usr/bin/env python3
"""
Preprocess large tree datasets into batched tensor files for memory-efficient training.

This script converts tree datasets into preprocessed tensor batches that can be loaded
incrementally during training, avoiding RAM exhaustion on large datasets.

Usage:
    python -m dpvt.preprocess_dataset \
        --trees-file path/to/trees.pkl \
        --labels-file path/to/labels.pkl \
        --output-dir /fh/fast/matsen_e/lcollien/dpvt-experiments-1/preprocessed/my_dataset \
        --batch-size 100 \
        --num-workers 4

The output directory will contain:
    - metadata.json: Dataset metadata (max dimensions, number of batches, etc.)
    - batch_0000.pt: First batch of trees (tensors for batch_size trees)
    - batch_0001.pt: Second batch
    - ...

You can then use this preprocessed data with LazyTraversalDataset during training.
"""

import argparse
import json
import pickle
import torch
from pathlib import Path
from typing import List, Tuple
import sys
from datetime import datetime

# DNA states
STATES = ["A", "G", "C", "T"]
STATE_TO_IDX = {"A": 0, "G": 1, "C": 2, "T": 3}


def load_data(trees_file: str, labels_file: str) -> Tuple[List, List]:
    """Load trees and labels from pickle files."""
    print(f"Loading data from {trees_file} and {labels_file}...")

    with open(trees_file, 'rb') as f:
        trees = pickle.load(f)

    with open(labels_file, 'rb') as f:
        labels = pickle.load(f)

    if len(trees) != len(labels):
        raise ValueError(f"Mismatch: {len(trees)} trees but {len(labels)} labels")

    print(f"  Loaded {len(trees)} trees")
    return trees, labels


def compute_dataset_dimensions(trees: List) -> dict:
    """Compute maximum dimensions across all trees."""
    print("Computing dataset dimensions...")

    max_n_sites = max([len(tree.sequence) for tree in trees])
    max_n_nodes = max([len(list(tree.traverse())) for tree in trees])
    max_n_int_nodes = max([len(tree) - 2 for tree in trees])

    print(f"  Max sites: {max_n_sites}")
    print(f"  Max nodes: {max_n_nodes}")
    print(f"  Max internal nodes: {max_n_int_nodes}")

    return {
        'max_n_sites': max_n_sites,
        'max_n_nodes': max_n_nodes,
        'max_n_int_nodes': max_n_int_nodes,
        'n_trees': len(trees),
    }


def process_single_tree(tree, max_n_sites: int, max_n_nodes: int, max_n_int_nodes: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a single tree to its tensor representation.

    Returns:
        traversal: Tensor of shape (2, max_n_int_nodes, 3)
        mutations: Tensor of shape (max_n_nodes, max_n_sites, 4)
    """
    # Initialize tensors for this tree
    mutations = torch.full((max_n_nodes, max_n_sites, 4), -1, dtype=torch.float32)
    traversal = torch.full((2, max_n_int_nodes, 3), -1, dtype=torch.float32)

    # Create node index mapping
    node_index_dict = {
        node: index
        for (node, index) in zip(
            tree.traverse("preorder"), range(len(list(tree.traverse())))
        )
    }

    # Fill traversal tensor - postorder for upward traversal
    node_in_tensor_index = 0
    for node in tree.traverse("postorder"):
        if not (node.is_leaf() or node.is_root() or node.up.is_root()):
            children = node.get_children()
            traversal[0, node_in_tensor_index, :] = torch.tensor(
                [
                    node_index_dict[children[0]],
                    node_index_dict[children[1]],
                    node_index_dict[node],
                ]
            )
            node_in_tensor_index += 1

    # Fill traversal tensor - preorder for downward traversal and mutations
    node_in_traversal_index = 0
    n_sites = len(tree.sequence)
    for node in tree.traverse("preorder"):
        # Fill downward traversal (preorder) entries
        if not (node.is_leaf() or node.is_root() or node.up.is_root()):
            traversal[1, node_in_traversal_index, :] = torch.tensor(
                [
                    node_index_dict[node.up],
                    node_index_dict[node.get_sisters()[0]],
                    node_index_dict[node],
                ]
            )
            node_in_traversal_index += 1

        # Fill mutations
        for site_index in range(n_sites):
            mutations[node_index_dict[node], site_index, :] = 0.0
            if node.up is None:  # node is root
                pass
            else:  # non-root node
                n_seq = node.sequence[site_index]
                p_seq = node.up.sequence[site_index]
                try:
                    mutations[node_index_dict[node], site_index, STATE_TO_IDX[n_seq]] += 1
                    mutations[node_index_dict[node], site_index, STATE_TO_IDX[p_seq]] -= 1
                except KeyError:
                    raise ValueError(f"Each node sequence must be in {STATES}")

    return traversal, mutations


def create_mask_for_tree(tree, max_n_nodes: int) -> torch.Tensor:
    """Create mask for pendant edges (True for internal edges, False for pendant)."""
    mask = torch.zeros(max_n_nodes, dtype=torch.bool)
    for idx, node in enumerate(tree.traverse("preorder")):
        mask[idx] = not (node.is_leaf() or node.is_root() or node.up.is_root())
    return mask


def process_labels_for_tree(labels: List, max_n_nodes: int) -> torch.Tensor:
    """Convert labels list to padded tensor."""
    label_tensor = torch.zeros(max_n_nodes, dtype=torch.float32)
    label_tensor[:len(labels)] = torch.tensor(labels, dtype=torch.float32)
    return label_tensor


def process_batch(
    trees: List,
    labels: List,
    batch_start: int,
    batch_size: int,
    max_n_sites: int,
    max_n_nodes: int,
    max_n_int_nodes: int,
) -> dict:
    """
    Process a batch of trees into tensors.

    Returns:
        Dictionary containing batched tensors for traversal, mutations, labels, and masks.
    """
    batch_end = min(batch_start + batch_size, len(trees))
    actual_batch_size = batch_end - batch_start

    # Allocate batch tensors
    batch_traversals = torch.full(
        (actual_batch_size, 2, max_n_int_nodes, 3), -1, dtype=torch.float32
    )
    batch_mutations = torch.full(
        (actual_batch_size, max_n_nodes, max_n_sites, 4), -1, dtype=torch.float32
    )
    batch_labels = torch.zeros((actual_batch_size, max_n_nodes), dtype=torch.float32)
    batch_masks = torch.zeros((actual_batch_size, max_n_nodes), dtype=torch.bool)

    # Process each tree in the batch
    for i, tree_idx in enumerate(range(batch_start, batch_end)):
        tree = trees[tree_idx]
        tree_labels = labels[tree_idx]

        # Process tree
        traversal, mutations = process_single_tree(
            tree, max_n_sites, max_n_nodes, max_n_int_nodes
        )
        mask = create_mask_for_tree(tree, max_n_nodes)
        label_tensor = process_labels_for_tree(tree_labels, max_n_nodes)

        # Add to batch
        batch_traversals[i] = traversal
        batch_mutations[i] = mutations
        batch_labels[i] = label_tensor
        batch_masks[i] = mask

    return {
        'traversal': batch_traversals,
        'mutations': batch_mutations,
        'labels': batch_labels,
        'masks': batch_masks,
    }


def preprocess_dataset(
    trees_file: str,
    labels_file: str,
    output_dir: str,
    batch_size: int = 100,
):
    """
    Main preprocessing function.

    Args:
        trees_file: Path to pickle file containing list of trees
        labels_file: Path to pickle file containing list of labels
        output_dir: Directory to save preprocessed batches
        batch_size: Number of trees per batch file
    """
    start_time = datetime.now()
    print("="*80)
    print("DPVT Dataset Preprocessing")
    print("="*80)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_path}")

    # Load data
    trees, labels = load_data(trees_file, labels_file)

    # Compute dimensions
    dimensions = compute_dataset_dimensions(trees)
    max_n_sites = dimensions['max_n_sites']
    max_n_nodes = dimensions['max_n_nodes']
    max_n_int_nodes = dimensions['max_n_int_nodes']
    n_trees = dimensions['n_trees']

    # Calculate number of batches
    n_batches = (n_trees + batch_size - 1) // batch_size
    print(f"\nProcessing {n_trees} trees into {n_batches} batches of size {batch_size}")

    # Estimate memory usage per batch
    batch_memory_mb = (
        batch_size * max_n_nodes * max_n_sites * 4 * 4 +  # mutations
        batch_size * 2 * max_n_int_nodes * 3 * 4  # traversal
    ) / (1024 ** 2)
    print(f"Estimated memory per batch: {batch_memory_mb:.2f} MB")

    # Process batches
    print("\nProcessing batches...")
    print("-" * 80)

    for batch_idx in range(n_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, n_trees)

        # Process batch
        batch_data = process_batch(
            trees, labels, batch_start, batch_size,
            max_n_sites, max_n_nodes, max_n_int_nodes
        )

        # Save batch
        batch_filename = output_path / f"batch_{batch_idx:04d}.pt"
        torch.save(batch_data, batch_filename)

        # Progress
        progress = (batch_idx + 1) / n_batches * 100
        print(f"  Batch {batch_idx:4d}/{n_batches}: Trees {batch_start:6d}-{batch_end:6d} | "
              f"Progress: {progress:5.1f}% | Saved to {batch_filename.name}")

    # Save metadata
    metadata = {
        'max_n_sites': max_n_sites,
        'max_n_nodes': max_n_nodes,
        'max_n_int_nodes': max_n_int_nodes,
        'n_trees': n_trees,
        'n_batches': n_batches,
        'batch_size': batch_size,
        'preprocessing_date': datetime.now().isoformat(),
        'trees_file': trees_file,
        'labels_file': labels_file,
    }

    metadata_file = output_path / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print("-" * 80)
    print(f"\nMetadata saved to {metadata_file}")
    print("\nPreprocessing complete!")

    elapsed = datetime.now() - start_time
    print(f"Total time: {elapsed}")
    print(f"Output directory: {output_path}")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess tree datasets for memory-efficient training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--trees-file',
        type=str,
        required=True,
        help='Path to pickle file containing list of trees'
    )

    parser.add_argument(
        '--labels-file',
        type=str,
        required=True,
        help='Path to pickle file containing list of labels'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Directory to save preprocessed batches'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Number of trees per batch file (default: 100)'
    )

    args = parser.parse_args()

    try:
        preprocess_dataset(
            trees_file=args.trees_file,
            labels_file=args.labels_file,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
        )
    except Exception as e:
        print(f"\nError during preprocessing: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


def preprocess_from_memory(
    trees: List,
    labels: List,
    output_dir: str,
    batch_size: int = 100,
):
    """
    Preprocess trees and labels that are already loaded in memory.

    This is a convenience function for when you have trees and labels
    already loaded, and don't want to save/load pickle files.

    Args:
        trees: List of tree objects
        labels: List of label lists
        output_dir: Directory to save preprocessed batches
        batch_size: Number of trees per batch file

    Example:
        from dpvt.preprocess_dataset import preprocess_from_memory

        # Assuming you have trees and labels loaded
        preprocess_from_memory(
            trees=my_trees,
            labels=my_labels,
            output_dir='/path/to/output',
            batch_size=100
        )
    """
    start_time = datetime.now()
    print("="*80)
    print("DPVT Dataset Preprocessing (from memory)")
    print("="*80)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_path}")

    # Validate input
    if len(trees) != len(labels):
        raise ValueError(f"Mismatch: {len(trees)} trees but {len(labels)} labels")

    print(f"Processing {len(trees)} trees")

    # Compute dimensions
    dimensions = compute_dataset_dimensions(trees)
    max_n_sites = dimensions['max_n_sites']
    max_n_nodes = dimensions['max_n_nodes']
    max_n_int_nodes = dimensions['max_n_int_nodes']
    n_trees = dimensions['n_trees']

    # Calculate number of batches
    n_batches = (n_trees + batch_size - 1) // batch_size
    print(f"\nProcessing {n_trees} trees into {n_batches} batches of size {batch_size}")

    # Estimate memory usage per batch
    batch_memory_mb = (
        batch_size * max_n_nodes * max_n_sites * 4 * 4 +  # mutations
        batch_size * 2 * max_n_int_nodes * 3 * 4  # traversal
    ) / (1024 ** 2)
    print(f"Estimated memory per batch: {batch_memory_mb:.2f} MB")

    # Process batches
    print("\nProcessing batches...")
    print("-" * 80)

    for batch_idx in range(n_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, n_trees)

        # Process batch
        batch_data = process_batch(
            trees, labels, batch_start, batch_size,
            max_n_sites, max_n_nodes, max_n_int_nodes
        )

        # Save batch
        batch_filename = output_path / f"batch_{batch_idx:04d}.pt"
        torch.save(batch_data, batch_filename)

        # Progress
        progress = (batch_idx + 1) / n_batches * 100
        print(f"  Batch {batch_idx:4d}/{n_batches}: Trees {batch_start:6d}-{batch_end:6d} | "
              f"Progress: {progress:5.1f}% | Saved to {batch_filename.name}")

    # Save metadata
    metadata = {
        'max_n_sites': max_n_sites,
        'max_n_nodes': max_n_nodes,
        'max_n_int_nodes': max_n_int_nodes,
        'n_trees': n_trees,
        'n_batches': n_batches,
        'batch_size': batch_size,
        'preprocessing_date': datetime.now().isoformat(),
        'source': 'memory',
    }

    metadata_file = output_path / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print("-" * 80)
    print(f"\nMetadata saved to {metadata_file}")
    print("\nPreprocessing complete!")

    elapsed = datetime.now() - start_time
    print(f"Total time: {elapsed}")
    print(f"Output directory: {output_path}")
    print("="*80)


if __name__ == '__main__':
    main()
