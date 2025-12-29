from ete3 import Tree

from dpvt.models import (
    TraverseNN,
    TraverseMaxPooling,
)


def assign_sequences(trees, seq_dict):
    """Assign sequences to all nodes in trees."""
    for tree in trees:
        for node in tree.traverse():
            if node.name in seq_dict:
                node.sequence = seq_dict[node.name]
            if node.is_leaf():
                node.node_id = node.name
            else:
                node.node_id = ""
    return trees


def create_good_trees():
    """Create test trees with 2-site sequences, rooted at outgroup."""
    # Tree rooted at outgroup s5, with root having single child
    tree1_nwk = "((((s1,s2)i1,s3)i2,s4)s5)s5;"
    tree1 = Tree(tree1_nwk, format=8)
    seq_dict = {
        "s1": "AA",
        "s2": "CA",
        "s3": "CG",
        "s4": "TG",
        "s5": "GG",
        "i1": "CA",
        "i2": "CG",
    }
    assign_sequences([tree1], seq_dict)
    return [tree1]


def create_site4_good_trees():
    """Create test trees with 4-site sequences, rooted at outgroup."""
    # Tree rooted at outgroup s5, with root having single child
    tree1_nwk = "((((s1,s2)i1,s3)i2,s4)s5)s5;"
    tree1 = Tree(tree1_nwk, format=8)
    seq_dict = {
        "s1": "AAAA",
        "s2": "CAAA",
        "s3": "CGAA",
        "s4": "TGAA",
        "s5": "GGAA",
        "i1": "CAAA",
        "i2": "CGAA",
    }
    assign_sequences([tree1], seq_dict)
    return [tree1]


good_trees = create_good_trees()
site4_good_trees = create_site4_good_trees()


def test_nn():
    model = TraverseNN()  # learning_rate=0.01

    tree = good_trees[0]
    out = model([tree])
    for x in out[0]:
        assert x.item() > 0

    tree = site4_good_trees[0]
    out = model([tree])
    for x in out[0]:
        assert x.item() > 0


def test_max_pooling_nn():
    model = TraverseMaxPooling()

    tree = good_trees[0]
    out = model([tree])
    for x in out[0]:
        assert x.item() > 0

    tree = site4_good_trees[0]
    out = model([tree])
    for x in out[0]:
        assert x.item() > 0
