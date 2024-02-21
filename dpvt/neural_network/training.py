import torch
from torch import nn
from torch import optim
from torch.utils.data import (
    random_split,
    dataset,
    DataLoader,
)
import lightning as L
import matplotlib.pyplot as plt

from dpvt.neural_network.traverse_nn import TraverseNN
from dpvt.neural_network.training_data import (
    good_trees,
    bad_trees,
)


# tnn = TraverseNN()
lr = 0.05
epochs = 40
n = 5

loss_fn = nn.BCEWithLogitsLoss(reduction="sum")


class FourLeafData(dataset.Dataset):
    def __init__(self):
        self.data = good_trees + bad_trees
        self.labels = [0.0 for _ in range(12)] + [1.0 for _ in range(12)]

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


train_data, test_data = random_split(FourLeafData(), [20, 4])
train_loader = DataLoader(train_data, batch_size=2, collate_fn=custom_collate)
test_loader = DataLoader(test_data, batch_size=2, collate_fn=custom_collate)

# use pytorch lightning
tnn = TraverseNN()
trainer = L.Trainer(max_epochs=100)
if __name__ == "__main__":
    trainer.fit(tnn, train_loader, test_loader)
