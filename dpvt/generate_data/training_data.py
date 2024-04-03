import random
import pickle
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
neutral_trees = nwk_list_to_trees(pattern_to_nwk_list(neutral_template))


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


def collate_sequences(trees):
    """returns a tree with sequences obtained from concatenating sequences from  the
    input trees. Assumes all input trees have same topology
    Args:
        trees (list of ete3 Trees)
    """
    # check that all input trees have same topology
    topology_nwk = trees[0].write(format=100)
    for tree in trees[1:]:
        assert topology_nwk == tree.write(
            format=100
        ), "trees have different topology, collation failed"
    # end check
    seq_lists = [[n.sequence for n in tree.traverse()] for tree in trees]
    tree = trees[0].copy()
    for i, n in enumerate(tree.traverse()):
        n.sequence = "".join([seq_list[i] for seq_list in seq_lists])
        n.name = n.sequence
    return tree


"""
4-site, 4-leaf trees
"""

# site4_nwk = "((0000,(1111,1111)1111)0000)0000;"
# site4_nwk = pattern_to_nwk_list(site4_nwk)[0]
# site4_tree = nwk_list_to_trees([site4_nwk])[0]

SAMPLE_SIZE = 240
assert SAMPLE_SIZE % 2 == 0
seed = 21032024

def create_site4_good_trees(n_trees=SAMPLE_SIZE // 2, seed=None):
    """
    generate "good" trees for training by concatenating 2 "good" sites with 1 "bad" and
    1 "neutral", shuffled in random site-order
    """
    if seed is not None:
        random.seed(seed)
    trees = []
    for _ in range(n_trees):
        t1 = random.choice(good_trees[:12])
        t2 = random.choice(good_trees[:12])
        t3 = random.choice(bad_trees[:12])
        t4 = random.choice(neutral_trees[:12])
        t1, t2, t3, t4 = random.sample([t1, t2, t3, t4], 4)
        tree = collate_sequences([t1, t2, t3, t4])
        assign_features(tree)
        trees.append(tree)
    return trees

def create_site4_bad_trees(n_trees=SAMPLE_SIZE // 2, seed=None):
    """
    generate "bad" trees for training by concatenating 2 "bad" sites with 1 "good" and
    1 "neutral", shuffled in random site-order
    """
    if seed is not None:
        random.seed(seed)
    trees = []
    for _ in range(n_trees):
        t1 = random.choice(bad_trees[:12])
        t2 = random.choice(bad_trees[:12])
        t3 = random.choice(good_trees[:12])
        t4 = random.choice(neutral_trees[:12])
        t1, t2, t3, t4 = random.sample([t1, t2, t3, t4], 4)
        tree = collate_sequences([t1, t2, t3, t4])
        assign_features(tree)
        trees.append(tree)
    return trees
    
site4_good_trees = create_site4_good_trees(seed=seed)
file_name = f"4site_4leaf_{SAMPLE_SIZE}good_trees_{seed}seed.pickle"
with open(file_name, "wb") as fh:
    pickle.dump(site4_good_trees, fh)

site4_bad_trees = create_site4_bad_trees(seed=seed)
file_name = f"4site_4leaf_{SAMPLE_SIZE}bad_trees_{seed}seed.pickle"
with open(file_name, "wb") as fh:
    pickle.dump(site4_bad_trees, fh)
