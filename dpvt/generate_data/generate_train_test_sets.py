import sys
import os
from pathlib import Path
import torch
from torch.utils.data import (
    random_split,
    dataset,
)

sys.path.append("..")

from generate_data.training_data import (
    good_trees,
    bad_trees,
    site4_good_trees,
    site4_bad_trees,
)

import pickle


class FourLeafData(dataset.Dataset):
    def __init__(self):
        self.data = good_trees + bad_trees
        self.labels = [0.0 for _ in range(24)] + [1.0 for _ in range(24)]

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)


class FourLeafFourSiteData(dataset.Dataset):
    def __init__(self):
        self.data = site4_good_trees + site4_bad_trees
        self.labels = [0.0 for _ in range(10)] + [1.0 for _ in range(10)]

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)


def custom_collate(items):
    """
    Args:
        items is a list of (input, output) pairs, where `input` is an ete3.Tree and
        `output` is a float
    """
    return [item[0] for item in items], torch.tensor([item[1] for item in items])


def create_training_data(file_path):
    train_data, test_data = random_split(FourLeafFourSiteData(), [16, 4])
    data_dict = {
        "train": train_data,
        "val": test_data
    }
    with open(file_path, "wb") as f:
        pickle.dump(data_dict, file = f)


def main():
    data_dir = Path(__file__).parent / "../data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    file_path = data_dir / "4leaf4site.p"
    create_training_data(file_path)


if __name__ == "__main__":
    main()