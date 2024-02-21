import random
from itertools import (
    combinations,
    permutations,
)

import torch
from ete3 import Tree

STATES = ["A", "G", "C", "T"]
STATE_TO_IDX = {"A": 0, "G": 1, "C": 2, "T": 3}


def assign_features(tree):
    """
    modifies input tree by adding a `to_parent` dict attribute, where
    `to_parent["feature_0"]` is a 4-element torch.tensor which records the mutation from
    the node's parent to the (child) node, e.g., a mutation `A -> T` is encoded as
    [-1, 0, 0, 1]
    Args:
        tree (ete3 Tree): has sequence attribute on each node
    Returns: None
    """
    for node in tree.traverse():
        mut_vec = [0, 0, 0, 0]
        if node.up is None:  # node is root
            pass
        else:  # non-root node
            try:
                n_seq = node.sequence
                p_seq = node.up.sequence
            except AttributeError:
                n_seq = node.name
                p_seq = node.up.name
            try:
                mut_vec[STATE_TO_IDX[n_seq]] += 1
                mut_vec[STATE_TO_IDX[p_seq]] -= 1
            except KeyError:
                raise ValueError(f"Each node sequence must be in {STATES}")
        try:
            node.to_parent["feature_0"] = torch.tensor(mut_vec)
        except AttributeError:
            node.add_feature("to_parent", {"feature_0": torch.tensor(mut_vec)})
    return None


def pattern_to_nwk_list(temp):
    nwks = [temp.replace("0", a).replace("1", b) for a, b in permutations(STATES, 2)]
    return nwks


good_template = "((0,(1,1)1)0)0;"
"""
   /-0
-0|
  |   /-1
   \1|
      \-1
"""
good_nwks = pattern_to_nwk_list(good_template)
# good_nwks = [
#     "(A,(G,G)G)A;",
#     "(A,(C,C)C)A;",
#     ...
# ]
bad_template = "((1,(1,0)0)0)0;"
"""
   /-1
-0|
  |   /-1
   \0|
      \-0
"""
bad_nwks = pattern_to_nwk_list(bad_template)
# bad_nwks = [
#     "(G,(G,A)A)A;",
#     "(C,(C,A)A)A;",
#     ...
# ]


def nwk_list_to_trees(nwks):
    """
    Takes a list of newick strings as input, and returns a list of corresponding trees,
    where each tree is annotated by appropriate information for use in neural network
    """
    trees = [Tree(nwk, format=8) for nwk in nwks]
    for tree in trees:
        for node in tree.traverse():
            node.sequence = node.name
        assign_features(tree)
    return trees


good_trees = nwk_list_to_trees(good_nwks)

bad_trees = nwk_list_to_trees(bad_nwks)

"""
convenience functions
"""


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
        x = node.to_parent["feature_0"]
        if torch.count_nonzero(x).item() > 0:
            count += 1
    return count


def leaf_state_count(tree):
    leaf_states = set(leaf.sequence for leaf in tree.get_leaves())
    return len(leaf_states)


def homoplasy_count(tree):
    return mutation_count(tree) - leaf_state_count(tree) + 1


def is_perfect(tree):
    n_mutations = mutation_count(tree)
    n_leaf_states = leaf_state_count(tree)
    return n_leaf_states == n_mutations + 1
