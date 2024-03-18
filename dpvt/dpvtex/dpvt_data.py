import pickle
from torch.utils.data import (
    Dataset,
)
from pathlib import Path

# Get the absolute path to the directory where the current script is located
script_directory = Path(__file__).resolve().parent

dataset_dict = {
    "FourLeafFourSite": script_directory.parent / "data/4leaf4site.p"
}


class TreeDataset(Dataset):
    def __init__(self, data_dict):
        self.data = list(data_dict.items())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item, label = self.data[idx]
        return item, label


def train_val_data_of_nicknames(data_name):
    file_path = dataset_dict[data_name]
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    train_data = TreeDataset(data["train"])
    val_data = TreeDataset(data["val"])
    return train_data, val_data