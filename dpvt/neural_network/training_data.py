import random
from itertools import combinations

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
            try:
                n_seq = node.sequence
                mut_vec[STATE_TO_IDX[n_seq]] += 1
                p_seq = node.up.sequence
                mut_vec[STATE_TO_IDX[p_seq]] -= 1
            except KeyError:
                raise NotImplementedError(f"Each node sequence must be in {STATES}")
            node.add_feature("feature_0", torch.tensor(mut_vec))
    return None



good_template = "(0,(1,1)1)0;"
good_nwks = [
    good_template.replace("0", a).replace("1", b) for a, b in combinations(STATES, 2)
]
# good_nwks = [
#     "(A,(G,G)G)A;",
#     "(A,(C,C)C)A;",
#     ...
# ]
bad_template = "(1,(1,0)0)0;"
bad_nwks = [bad_template.replace("0", a).replace("1", b) for a, b in combinations(STATES, 2)]
# bad_nwks = [
#     "(G,(G,A)A)A;",
#     "(C,(C,A)A)A;",
#     ...
# ]

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

good_test_trees = [
    Tree(good_template.replace("0", b).replace("1", a), format=8) 
    for a, b in combinations(STATES, 2)
]
bad_test_trees = [
    Tree(bad_template.replace("0", b).replace("1", a), format=8) 
    for a, b in combinations(STATES, 2)
]
for tree in good_test_trees + bad_test_trees:
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
    """returns a new tree which has the same topology as the input tree, but with 
    random node sequences and names"""
    tree = tree.copy()
    for node in tree:
        s = random.choice(STATES)
        node.name = s
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
    leaf_states = set(leaf.sequence for leaf in tree.get_leaves())
    return len(leaf_states)

def is_perfect(tree):
    n_mutations = mutation_count(tree)
    n_leaf_states = leaf_state_count(tree)
    return n_leaf_states == n_mutations + 1