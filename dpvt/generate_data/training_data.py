import random
import pickle
from itertools import (
    combinations,
    permutations,
)

import torch
from ete3 import Tree
from dpvt.neural_network.models import TraverseNN
from dpvt.generate_data.utils import Tree as MyTree
from dpvt.generate_data.perfect_phylogeny import PerfectPhylogeny
from dpvt.generate_data.perturb_phylogeny import (
    perturb_tree,
    sankoff_for_missing_sequences,
)

STATES = ["A", "G", "C", "T"]
STATE_TO_IDX = {"A": 0, "G": 1, "C": 2, "T": 3}


assign_features = TraverseNN.assign_mutation_vectors

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
        x = node.to_parent["edge_mutation"]
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
        trees.append(tree)
    return trees

site4_good_trees = create_site4_good_trees(seed=seed)
site4_bad_trees = create_site4_bad_trees(seed=seed)


"""
trees from perturbing perfect phylogenies
"""


def create_mixed_perfect_phylo_trees(n_leaves, n_trees, n_phylos_per_tree,):
    """
    Create a collection of phylogenies obtained by randomly mixing a subtree of a
    perfect phylogeny.
    """
    DEPTH = 3
    # n_phylos_per_tree = 32
    tree_data_dict = {}
    for _ in range(n_trees):
        tree = MyTree()
        tree.populate(n_leaves, model="uniform")
        pp = PerfectPhylogeny(tree)
        for _ in range(n_phylos_per_tree):
            phylo = pp.make_random_phylogeny()
            mixed_phylo = perturb_tree(phylo, depth=DEPTH, exception_on_fail=True)
            # assert(mixed_phylo is not None)
            sankoff_for_missing_sequences(mixed_phylo)
            # convert custom Tree object to ete3 Tree
            newick = mixed_phylo.write(
                features=["sequence", "random_tree"], 
                format_root_node=True
            )
            mixed_phylo = Tree(newick)
            # add "extra" unifurcating root above previous root
            new_tree = Tree()
            new_tree.add_child(mixed_phylo)
            new_tree.sequence = mixed_phylo.sequence
            new_tree.random_tree = mixed_phylo.random_tree
            mixed_phylo = new_tree
            edge_classifier = [
                1.0 if (node.random_tree and not node.is_leaf()) else 0.0
                for node in mixed_phylo.traverse(strategy="preorder")
            ]
            tree_data_dict[mixed_phylo] = edge_classifier
    return tree_data_dict