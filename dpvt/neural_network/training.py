import torch
from torch import nn
from torch import optim
import matplotlib.pyplot as plt

from dpvt.neural_network.traverse_nn import TraverseNN
from dpvt.neural_network.training_data import (
    good_trees, bad_trees,
    good_test_trees, bad_test_trees,
)


# tnn = TraverseNN()
lr = 0.05
epochs = 20
n = 5

loss_fn = nn.BCEWithLogitsLoss(reduction="sum")

train_data = list(
    zip(good_trees + good_test_trees[:-1], bad_trees + bad_test_trees[:-1])
)
train_in = good_trees + good_test_trees[:-1] + bad_trees + bad_test_trees[:-1]
train_out = [0. for _ in range(11)] + [1. for _ in range(11)]

def get_model():
    model = TraverseNN()
    return model, optim.SGD(model.parameters(), lr=lr)

tnn, opt = get_model()
# print("Untrained loss:", loss_fn(tnn(train_in[:2]), torch.tensor(train_out[:2])))

# training loop
def fit(verbose=True, log_out=True):
    # set tnn to train mode
    tnn.train()
    log = []
    for ep in range(epochs):
        for i, xb in enumerate(train_data):
            opt.zero_grad()
            # xb = [good_trees[i], bad_trees[i]]
            yb = torch.tensor([0.0, 1.0])

            # compute prediction and loss
            pred = tnn(xb)
            loss = loss_fn(pred, yb)
            if verbose and i == 0:
                print("prediction:", pred.tolist())
                print("loss:", loss.item())
            if log_out:
                log.append(loss.item())

            loss.backward()
            opt.step()
        if verbose: print(f"end epoch {ep}")
    if log_out: return log
# fit()

def fit_and_plot(out_file="test.pdf"):
    log = fit()
    fig, ax = plt.subplots()
    ax.plot(log)
    fig.savefig(out_file)


def run():
    pass