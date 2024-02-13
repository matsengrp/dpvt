import torch
from torch import nn
from torch import optim
from ete3 import Tree
from traversal_nn import TraverseNN
from training_data import assign_features, good_trees, bad_trees


# tnn = TraverseNN()
lr = 0.5
opt_lr = 0.5
epochs = 2
n = 5

loss_fn = nn.BCEWithLogitsLoss()
# loss_fn = nn.MSELoss()

def get_model():
    model = TraverseNN()
    return model, optim.SGD(model.parameters(), lr=opt_lr)

tnn, opt = get_model()
# print(loss_fn(tnn(xb), yb))

# training loop
def fit():
    # set tnn to train mode
    tnn.train()
    for _ in range(epochs):
        for _ in range(n):
            opt.zero_grad()
            xb = [t_good, t_bad]
            yb = torch.tensor([0.0, 1.0], requires_grad=True)
            # yb = torch.tensor([[-100.0], [100.0]], requires_grad=True)

            # compute prediction and loss
            pred = tnn(xb)
            loss = loss_fn(pred, yb)
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
# fit()

def fit_no_batch():
    # set tnn to train mode
    tnn.train()
    for _ in range(epochs):
        for _ in range(n):
            y_good, y_bad = (torch.tensor([x]) for x in [0.0, 1.0])

            # opt.zero_grad()
            loss = 0
            for tree, y in ((t_good, y_good), (t_bad, y_bad)):
                # compute prediction and loss
                pred = tnn(tree)
                loss += loss_fn(pred, y)
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
                        pass
                tnn.zero_grad()


"""
tree data for training
"""
# tree with maximum parsimony
t_good = Tree("(0,(1,2));")
for node in (t_good, t_good.children[0]):
    node.sequence = "A"
for node in (
    t_good.children[1], t_good.children[1].children[0], t_good.children[1].children[1]
):
    node.sequence = "T"
"""
   /-A
-A|
  |   /-T
   \T|
      \-T
"""# tree without maximum parsimony
t_bad = Tree("(0,(1,2));")
for node in (t_bad, t_bad.children[1], t_bad.children[1].children[1]):
    node.sequence = "A"
for node in (t_bad.children[0], t_bad.children[1].children[0]):
    node.sequence = "T"
"""
   /-T
-A|
  |   /-T
   \A|
      \-A
"""
# assign mutation features
for tree in [t_good, t_bad]:
    assign_features(tree)

# print("good tree:")
# print(t_good.get_ascii(attributes=["sequence"]))
# print(t_good.get_ascii(attributes=["feature_0"]))

# print("\n")
# print("bad tree:")
# print(t_bad.get_ascii(attributes=["sequence"]))
# print(t_bad.get_ascii(attributes=["feature_0"]))
