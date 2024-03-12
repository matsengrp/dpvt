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
    `to_parent["feature_0"]` is a m-by-4 torch.tensor which records the mutation from
    the node's parent to the (child) node, e.g., a mutation `A -> T` is encoded as
    [...,[-1, 0, 0, 1],...]
    Args:
        tree (ete3 Tree): has sequence attribute on each node
    Returns: None
    """
    n_sites = len(tree.sequence)
    for node in tree.traverse():
        for i in range(n_sites):
            mut_vec = [0.0, 0.0, 0.0, 0.0]
            if node.up is None:  # node is root
                pass
            else:  # non-root node
                n_seq = node.sequence[i]
                p_seq = node.up.sequence[i]
                # except AttributeError:
                #     n_seq = node.name
                #     p_seq = node.up.name
                try:
                    mut_vec[STATE_TO_IDX[n_seq]] += 1
                    mut_vec[STATE_TO_IDX[p_seq]] -= 1
                except KeyError:
                    raise ValueError(f"Each node sequence must be in {STATES}")
            new_row = torch.tensor(mut_vec).unsqueeze(0)
            if i == 0:
                node.add_feature("to_parent", {"feature_0": new_row})
            else:
                node.to_parent["feature_0"] = torch.cat(
                    (node.to_parent["feature_0"], new_row)
                )
            # try:
            #     node.to_parent["feature_0"] = torch.cat(
            #         (node.to_parent["feature_0"], new_row)
            #     )
            # except AttributeError:
            #     node.add_feature("to_parent", {"feature_0": new_row})
    return None


def pattern_to_nwk_list(temp):
    """
    Takes a template newick string, containing 0's and 1's, and replaces these states
    with DNA basepair states
    """
    nwks = [temp.replace("0", a).replace("1", b) for a, b in permutations(STATES, 2)]
    return nwks

def pattern_to_nwk_random(temp):
    """
    Takes a template newick string, containing 0's and 1's, and replaces these states
    with a random choice of DNA basepair states
    """
    a, b = random.choice(list(permutations(STATES, 2)))
    return temp.replace("0", a).replace("1", b)

def nwk_to_tree(nwk):
    tree = Tree(nwk, format=8)
    for node in tree.traverse():
        node.sequence = node.name
    return tree

def nwk_list_to_trees(nwks):
    """
    Takes a list of newick strings as input, and returns a list of corresponding trees,
    where each tree is annotated by appropriate information for use in neural network
    """
    trees = [Tree(nwk, format=8) for nwk in nwks]
    trees += [reflect_tree(tree) for tree in trees]
    for tree in trees:
        for node in tree.traverse():
            node.sequence = node.name
        assign_features(tree)
    return trees

def reflect_tree(tree):
    """returns a new tree which has the same topology as the input tree, 
    but reflected by swapping the branches at each bifurcating internal node.
    """
    reflected_tree = tree.copy()
    for node in reflected_tree.traverse():
        if len(node.get_children()) == 2:
            node.children[0], node.children[1] = node.children[1], node.children[0]
    return reflected_tree

"""
1-site, 4-leaf trees
"""
good_template = "((0,(1,1)1)0)0;"
"""
      /-0
-0/-0|
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
-0/-0|
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


good_trees = nwk_list_to_trees(good_nwks)

bad_trees = nwk_list_to_trees(bad_nwks)

neutral_template = "((0,(1,0)0)0)0;"
"""
      /-0
-0/-0|
     |   /-1
      \0|
         \-0
"""


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

def collate_sequences(tree1, tree2):
    """returns a tree with sequences obtained from concatenating sequences from tree1
    and tree2"""
    seq_list_1 = [n.sequence for n in tree1.traverse()]
    seq_list_2 = [n.sequence for n in tree2.traverse()]
    tree = tree1.copy()
    for i, n in enumerate(tree.traverse()):
        n.sequence = seq_list_1[i] + seq_list_2[i]
        n.name = n.sequence
    return tree


"""
4-site, 4-leaf trees
"""

site4_nwk = "((0000,(1111,1111)1111)0000)0000;"
site4_nwk = pattern_to_nwk_list(site4_nwk)[0]
site4_tree = nwk_list_to_trees([site4_nwk])[0]

SAMPLE_SIZE = 20
site4_good_trees = []
for _ in range(SAMPLE_SIZE):
    t1 = nwk_to_tree(pattern_to_nwk_random(good_template))
    t2 = nwk_to_tree(pattern_to_nwk_random(good_template))
    t3 = nwk_to_tree(pattern_to_nwk_random(bad_template))
    t4 = nwk_to_tree(pattern_to_nwk_random(neutral_template))
    t1, t2, t3, t4 = random.sample([t1, t2, t3, t4], 4)
    t12 = collate_sequences(t1, t2)
    t34 = collate_sequences(t3, t4)
    tree = collate_sequences(t12, t34)
    assign_features(tree)
    site4_good_trees.append(tree)

site4_bad_trees = []
for _ in range(SAMPLE_SIZE):
    t1 = nwk_to_tree(pattern_to_nwk_random(good_template))
    t2 = nwk_to_tree(pattern_to_nwk_random(bad_template))
    t3 = nwk_to_tree(pattern_to_nwk_random(bad_template))
    t4 = nwk_to_tree(pattern_to_nwk_random(neutral_template))
    t1, t2, t3, t4 = random.sample([t1, t2, t3, t4], 4)
    t12 = collate_sequences(t1, t2)
    t34 = collate_sequences(t3, t4)
    tree = collate_sequences(t12, t34)
    assign_features(tree)
    site4_bad_trees.append(tree)
