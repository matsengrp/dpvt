import pytest
import torch
from ete3 import Tree

from dpvt.models import (
    TraverseNN,
    TraverseMaxPooling,
    TraverseAvgPooling,
)
from dpvt.wrapper import TraversalDataset


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


@pytest.mark.parametrize("model_cls", [TraverseNN, TraverseMaxPooling, TraverseAvgPooling])
@pytest.mark.parametrize("trees_fixture", ["good_trees", "site4_good_trees"])
def test_traversal_paths_agree(model_cls, trees_fixture):
    """forward_on_tree and forward_on_traversal must produce identical logits."""
    trees = good_trees if trees_fixture == "good_trees" else site4_good_trees
    tree = trees[0]
    n_nodes = len(list(tree.traverse()))
    labels = [[0] * n_nodes]

    # Match the real training environment: configure_torch() sets float64 as default.
    # TraversalDataset hardcodes float64; forward_on_tree creates tensors in the
    # default dtype. Both must agree.
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        model = model_cls()
        model.eval()
        max_seq_length = len(tree.sequence)

        dataset = TraversalDataset([tree], labels, device="cpu")
        traversal = dataset.traversal[0]
        mutations = dataset.mutations[0]

        with torch.no_grad():
            out_tree = model.forward_on_tree(tree, max_seq_length)
            out_traversal = model.forward_on_traversal(traversal, mutations)
    finally:
        torch.set_default_dtype(prev_dtype)

    assert out_tree.shape == out_traversal.shape, (
        f"Shape mismatch: forward_on_tree={out_tree.shape}, "
        f"forward_on_traversal={out_traversal.shape}"
    )
    assert torch.allclose(out_tree, out_traversal, atol=1e-6), (
        f"Max absolute difference: {(out_tree - out_traversal).abs().max().item()}"
    )
