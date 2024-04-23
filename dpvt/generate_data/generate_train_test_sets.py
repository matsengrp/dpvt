import os
import sys
from pathlib import Path
import torch
import pickle
from ete3 import Tree

sys.path.append("..")

from generate_data.training_data import (
    good_trees,
    bad_trees,
    site4_good_trees,
    site4_bad_trees,
    create_mixed_perfect_phylo_trees,
)


def create_training_data_toy(file_path, good_trees, bad_trees):
    """
    each label is a list of values corresponding to edges in the following tree
    topology, ordered by preorder traversal:

          /-2
    -0/-1|
         |   /-4
          \3|
             \-5
    """
    tree_to_label = {
        **{tree: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for tree in good_trees},
        **{tree: [0.0, 0.0, 0.0, 1.0, 0.0, 0.0] for tree in bad_trees},
    }

    # data_dict = {"train": train_data, "val": val_data}
    with open(file_path, "wb") as f:
        pickle.dump(tree_to_label, file=f)


def create_training_data_perfect(file_path, n_leaves, n_trees, n_phylos_per_tree,):
    """
    Create a collection of phylogenies obtained by randomly mixing a subtree of a
    perfect phylogeny.
    """

    tree_data_dict = create_mixed_perfect_phylo_trees(
        n_leaves=n_leaves,
        n_trees=n_trees,
        n_phylos_per_tree=n_phylos_per_tree,
    )

    with open(file_path, "wb") as fh:
        pickle.dump(tree_data_dict, file=fh)


def main():
    data_dir = Path(__file__).parent / "../data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    four_site_file_path = data_dir / "4leaf4site.p"
    if not os.path.exists(four_site_file_path):
        print(f"Creating dataset {four_site_file_path}")
        create_training_data_toy(four_site_file_path, site4_good_trees, site4_bad_trees)

    one_site_file_path = data_dir / "4leaf.p"
    if not os.path.exists(one_site_file_path):
        print(f"Creating dataset {one_site_file_path}")
        create_training_data_toy(one_site_file_path, good_trees, bad_trees)

    ten_leaf_file_path = data_dir / "10leaf_perfect.p"
    if not os.path.exists(ten_leaf_file_path):
        print(f"Creating dataset {ten_leaf_file_path}")
        create_training_data_perfect(
            ten_leaf_file_path, 
            n_leaves=10, 
            n_trees=32, 
            n_phylos_per_tree = 32,
        )
    thirty_leaf_file_path = data_dir / "30leaf_perfect.p"
    if not os.path.exists(thirty_leaf_file_path):
        print(f"Creating dataset {thirty_leaf_file_path}")
        create_training_data_perfect(
            thirty_leaf_file_path, 
            n_leaves=30, 
            n_trees=32, 
            n_phylos_per_tree=32,
        )

if __name__ == "__main__":
    main()
