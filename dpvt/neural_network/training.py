import torch
from torch import nn
from torch import optim
from torch.utils.data import (
    random_split,
    dataset,
    Dataset,
    DataLoader,
)
import matplotlib.pyplot as plt

from dpvt.neural_network.traverse_nn import TraverseNN
from dpvt.neural_network.training_data import (
    good_trees,
    bad_trees,
)


# tnn = TraverseNN()
lr = 0.05
epochs = 20
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


# four_leaf_data = Dataset.from_dict({
#     "data": good_trees + bad_trees,
#     "label": [0.0 for _ in range(12)] + [1.0 for _ in range(12)]
# })


def custom_collate(items):
    # print("debugging:", items)
    return [item[0] for item in items], torch.tensor([item[1] for item in items])


train_data, test_data = random_split(FourLeafData(), [20, 4])
train_loader = DataLoader(train_data, batch_size=2, collate_fn=custom_collate)
test_loader = DataLoader(test_data, batch_size=2, collate_fn=custom_collate)


def get_model():
    model = TraverseNN()
    return model, optim.SGD(model.parameters(), lr=lr)


tnn, opt = get_model()
print(
    "Untrained loss:",
    loss_fn(
        tnn(train_data[i][0] for i in range(5)),
        torch.tensor([train_data[i][1] for i in range(5)]),
    ),
)


# training loop
def fit(verbose=True, log_out=True):
    # set tnn to train mode
    log = []
    for ep in range(epochs):
        tnn.train()
        for xb, yb in train_loader:
            opt.zero_grad()

            # compute prediction and loss
            pred = tnn(xb)
            loss = loss_fn(pred, yb)
            # if verbose and i == 0:
            #     print("prediction:", pred.tolist())
            #     print("loss:", loss.item())
            if log_out:
                log.append(loss.item())

            loss.backward()
            opt.step()
        # validation step
        tnn.eval()
        with torch.no_grad():
            valid_loss = sum(loss_fn(tnn(xb), yb) for xb, yb in test_loader)
            print("validation loss:", valid_loss)
        if verbose:
            print(f"end epoch {ep + 1}")
    if log_out:
        return log


# fit()


def fit_and_plot(out_file="test.pdf"):
    log = fit()
    fig, ax = plt.subplots()
    ax.plot(log)
    fig.savefig(out_file)


def run():
    pass
