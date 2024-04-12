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

    data_dict = {
        **{tree: 0.0 for tree in good_trees},
        **{tree: 1.0 for tree in bad_trees},
    }
    # data_dict = {"train": train_data, "val": val_data}
    with open(file_path, "wb") as f:
        pickle.dump(data_dict, file=f)


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
