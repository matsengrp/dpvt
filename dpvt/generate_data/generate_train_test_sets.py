import os
import sys
from pathlib import Path
import pickle

sys.path.append("..")

from generate_data.training_data import (
    good_trees,
    bad_trees,
    site4_good_trees,
    site4_bad_trees,
)


def create_training_data(file_path, good_trees, bad_trees):
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


def main():
    data_dir = Path(__file__).parent / "../data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    four_site_file_path = data_dir / "4leaf4site.p"
    create_training_data(four_site_file_path, site4_good_trees, site4_bad_trees)

    one_site_file_path = data_dir / "4leaf.p"
    create_training_data(one_site_file_path, good_trees, bad_trees)


if __name__ == "__main__":
    main()
