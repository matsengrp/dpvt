import pickle
from sklearn.model_selection import train_test_split
from torch.utils.data import (
    Dataset,
)
import numpy as np
from pathlib import Path

# Get the absolute path to the directory where the current script is located
script_directory = Path(__file__).resolve().parent

dataset_dict = {
    "FourLeafFourSite": script_directory.parent / "data/4leaf4site.p",
    "FourLeaf": script_directory.parent / "data/4leaf.p",
    "TenLeaf": script_directory.parent / "data/10leaf_perfect.p",
    "ThirtyLeaf": script_directory.parent / "data/30leaf_perfect.p",
}


class TreeDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def train_val_data_of_nicknames(data_name):
    file_path = dataset_dict[data_name]
    with open(file_path, "rb") as f:
        data_dict = pickle.load(f)

    # split into 60% training, 20% validation, 20% testing
    train_size = int(0.6 * len(data_dict))
    val_size = int(0.2 * len(data_dict))
    test_size = len(data_dict) - train_size - val_size

    # Split into balanced training, validation, and test data using sklearn
    labels = list(data_dict.values())
    trees = list(data_dict.keys())

    # labels_array = np.array(labels)
    n_bad_edges = np.sum(labels, axis=1)

    train_val_data, test_data, train_val_labels, test_labels, sum_train_val, _ = (
        train_test_split(
            trees, labels, n_bad_edges, test_size=test_size, stratify=n_bad_edges
        )
    )
    train_data, val_data, train_labels, val_labels = train_test_split(
        train_val_data,
        train_val_labels,
        test_size=val_size / (train_size + val_size),
        stratify=sum_train_val,
    )

    train_data = TreeDataset(train_data, train_labels)
    test_data = TreeDataset(test_data, test_labels)
    val_data = TreeDataset(val_data, val_labels)
    return train_data, val_data, test_data
