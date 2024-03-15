import pickle

from generate_data.generate_train_test_sets import FourLeafFourSiteData

dataset_dict = {
    "FourLeafFourSite": "../data/4leaf4site.p"
}

def train_val_data_of_nicknames(data_name):
    file_path = dataset_dict[data_name]
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    train_data = data["train"]
    val_data = data["val"]
    return train_data, val_data