import torch
from torch import nn
from ete3 import Tree
from traversal_nn import TraverseNN

STATE_TO_IDX = {
    "A": 0,
    "G": 1,
    "C": 2,
    "T": 3
}

def assign_features(tree):
    """
    modifies input tree by adding attribute feature_0, which is a 4-element torch.tensor 
    which records the mutation from the parent to child node,
    e.g., a mutation `A -> T` is encoded as [-1, 0, 0, 1]
    Args:
        tree (ete3 Tree): has sequence attribute on each node
    Returns: None 
    """
    for node in tree.traverse():
        if node.up is None:
            # node is root
            node.add_feature("feature_0", torch.zeros(4))
        else:
            # non-root node
            mut_vec = [0, 0, 0, 0]
            n_seq = node.sequence
            mut_vec[STATE_TO_IDX[n_seq]] += 1
            p_seq = node.up.sequence
            mut_vec[STATE_TO_IDX[p_seq]] -= 1
            node.add_feature("feature_0", torch.tensor(mut_vec))
    return None



tnn = TraverseNN()
lr = 0.5
epochs = 2
n = 5

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
# fit()

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
"""
good_nwks = [
    "(A,(T,T)T)A;",
    "(G,(T,T)T)G;",
    "(C,(T,T)T)C;",
    "(C,(A,A)A)C;",
    "((T,T)T,G)G;",
    "((T,T)T,C)C;",
    "((G,G)G,A)A;",
    "((C,C)C,A)A;",
]
bad_nwks = [
    "(T,(T,A)A)A;",
    "(C,(C,A)A)A;",
    "(G,(G,A)A)A;",
    "(T,(T,A)A)A;",
    "(T,(T,A)A)A;",
    "(T,(T,A)A)A;",
]

good_trees = [Tree(nwk, format=8) for nwk in good_nwks]
for tree in good_trees:
    for n in tree.traverse():
        n.sequence = n.name
    assign_features(tree)

# tree without maximum parsimony
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

"""
      /-G
   /G|
-A|   \-G
  |
   \-A
"""
nwk = (
    "((0[&&NHX:sequence=G],1[&&NHX:sequence=G])[&&NHX:sequence=G],2[&&NHX:sequence=A])["
    "&&NHX:sequence=A];"
)
test_good = Tree(nwk)
assign_features(test_good)


nwk = (
    "((0[&&NHX:sequence=G],1[&&NHX:sequence=A])[&&NHX:sequence=G],2[&&NHX:sequence=A])["
    "&&NHX:sequence=G];"
)
"""
      /-G
   /G|
-G|   \-A
  |
   \-A
"""
test_bad = Tree(nwk)
assign_features(test_bad)
