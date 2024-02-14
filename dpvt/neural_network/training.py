import torch
from torch import nn
from torch import optim
from ete3 import Tree
from traversal_nn import TraverseNN
from training_data import (
    assign_features, 
    good_trees, bad_trees,
    good_test_trees, bad_test_trees,
)


# tnn = TraverseNN()
lr = 0.05
opt_lr = 0.05
epochs = 20
n = 5

loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
# loss_fn = nn.MSELoss()

def get_model():
    model = TraverseNN()
    return model, optim.SGD(model.parameters(), lr=opt_lr)

tnn, opt = get_model()
# print(loss_fn(tnn(xb), yb))

train_data = list(
    zip(good_trees + good_test_trees[:-1], bad_trees + bad_test_trees[:-1])
)

# training loop
def fit():
    # set tnn to train mode
    tnn.train()
    for ep in range(epochs):
        for i, xb in enumerate(train_data):
            opt.zero_grad()
            # xb = [good_trees[i], bad_trees[i]]
            yb = torch.tensor([0.0, 1.0], requires_grad=True)

            # compute prediction and loss
            pred = tnn(xb)
            loss = loss_fn(pred, yb)
            if i == 0:
                print("prediction:", pred.tolist())
                print("loss:", loss.item())

            loss.backward()
            opt.step()
            # # without torch.opt
            # with torch.no_grad():
            #     for p in tnn.parameters():
            #         # print(p.shape)
            #         try:
            #             p -= p.grad * lr
            #             # print(f"updated param, size {p.shape}")
            #         except TypeError as e:
            #             print(f"no update of param, size {p.shape};", e)
            #             pass
            #         tnn.zero_grad()
        print(f"end epoch {ep}")
# fit()

def fit_opt_no_batch():
    # set tnn to train mode
    tnn.train()
    for ep in range(epochs):
        for i, (good, bad) in enumerate(train_data):
            y_good, y_bad = (torch.tensor([x]) for x in [0.0, 1.0])

            opt.zero_grad()
            loss = 0
            for tree, y in ((good, y_good), (bad, y_bad)):
                # compute prediction and loss
                pred = tnn(tree)
                loss += loss_fn(pred, y)
                if i == 0:
                    print("target pred vs pred:", y.item(), ",", pred.item())
                    print("loss:", loss.item())

            loss.backward()
            opt.step()
        print(f"end epoch {ep}")

def fit_no_opt():
    # set tnn to train mode
    tnn.train()
    for ep in range(epochs):
        for i, (good, bad) in enumerate(train_data):
            y_good, y_bad = (torch.tensor([x]) for x in [0.0, 1.0])

            # opt.zero_grad()
            loss = 0
            for tree, y in ((good, y_good), (bad, y_bad)):
                # compute prediction and loss
                pred = tnn(tree)
                loss += loss_fn(pred, y)
                if i == 0:
                    print("target pred vs pred:", y.item(), ",", pred.item())
                    print("loss:", loss.item())

            loss.backward()
            # opt.step()
            with torch.no_grad():
                for p in tnn.parameters():
                    # print(p.shape)
                    try:
                        p -= p.grad * lr
                        # print(f"updated param, size {p.shape}")
                    except TypeError as e:
                        # print(f"no update of param, size {p.shape};", e)
                        raise e
            tnn.zero_grad()
        print(f"end epoch {ep}")

def run():
    pass