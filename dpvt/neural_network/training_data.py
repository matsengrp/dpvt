import random
import torch
from ete3 import Tree

STATES = ["A", "G", "C", "T"]
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
    for node in tree.traverse():
        node.sequence = node.name
    assign_features(tree)
bad_trees = [Tree(nwk, format=8) for nwk in bad_nwks]
for tree in bad_trees:
    for node in tree.traverse():
        node.sequence = node.name
    assign_features(tree)


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


def random_state_assignment(tree):
    tree = tree.copy()
    for node in tree:
        s = random.choice(STATES)
        node.sequence = s
    return tree

def mutation_count(tree):
    assign_features(tree)
    count = 0
    for node in tree.traverse():
        x = node.feature_0
        if torch.count_nonzero(x).item() > 0:
            count += 1
    return count

def leaf_state_count(tree):
    pass