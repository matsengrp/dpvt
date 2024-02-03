from ete3 import Tree
import torch
from torch import nn
from traversal_nn import TraverseNN

# tree with maximum parsimony
t_good = Tree("(0,(1,2));")
for node in (t_good, t_good.children[0]):
    node.sequence = "A"
for node in (t_good.children[1], t_good.children[1].children[0], t_good.children[1].children[1]):
    node.sequence = "T"

# tree without maximum parsimony
t_bad = Tree("(0,(1,2));")
for node in (t_bad, t_bad.children[1], t_bad.children[1].children[1]):
    node.sequence = "A"
for node in (t_bad.children[0], t_bad.children[1].children[0]):
    node.sequence = "T"

# assign mutation features
for tree in [t_good, t_bad]:
    for node in tree.traverse():
        if node.up is None:
            node.add_feature("feature_0", torch.zeros(4))
            continue
        n_seq = node.sequence
        p_seq = node.up.sequence
        if n_seq == p_seq:
            node.add_feature("feature_0", torch.zeros(4))
        else: # mutation is A -> T
            node.add_feature("feature_0", torch.tensor([-1,0,0,1]))

print("good tree:")
print(t_good.get_ascii(attributes=["sequence"]))
print(t_good.get_ascii(attributes=["feature_0"]))

print("\nbad tree:")
print(t_bad.get_ascii(attributes=["sequence"]))
print(t_bad.get_ascii(attributes=["feature_0"]))


tnn = TraverseNN()
lr = 0.5
epochs = 2
n = 20

loss_fn = nn.BCEWithLogitsLoss()

# training loop
def fit():
    for _ in range(epochs):
        for _ in range(n):
            # xb = [t_good, t_bad]
            # yb = torch.tensor([0.0, 1.0])
            # pred = torch.tensor(
            #     [tnn(t_good), tnn(t_bad)], requires_grad=True
            # )
            y_good = torch.tensor([-100.0])
            y_bad = torch.tensor([100.0])
            for tree, y in ((t_good, y_good), (t_bad, y_bad)):
                pred = tnn(tree)
                # loss = tnn.loss(pred, yb)
                loss = loss_fn(pred, y)
                print("prediction:", pred.item())
                print("loss:", loss.item())

                loss.backward()
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


fit()

