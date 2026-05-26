import csv
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

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

    # TraversalDataset hardcodes float32 for mutations and traversal tensors.
    # forward_on_tree creates tensors via torch.zeros using the default dtype.
    # Both must agree, so set the default to float32.
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
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
    assert torch.allclose(out_tree, out_traversal, atol=1e-4), (
        f"Max absolute difference: {(out_tree - out_traversal).abs().max().item()}"
    )


@pytest.mark.parametrize("model_cls", [TraverseNN, TraverseAvgPooling])
def test_on_test_epoch_end_saves_pr_curve(model_cls, tmp_path):
    """on_test_epoch_end writes pr_curve.pdf and pr_curve.csv and resets metrics."""
    model = model_cls()

    # Simulate what test_step accumulates for 4 samples.
    # test_targets stores [N,1] int targets (matching test_step's unsqueeze(-1));
    # metrics are updated with 1D tensors as torchmetrics requires matching shapes.
    logits = torch.tensor([2.0, 1.5, -1.0, -2.0], dtype=torch.float32)
    labels = torch.tensor([1, 1, 0, 0], dtype=torch.int32)
    probs = torch.sigmoid(logits)

    model.test_probs.append(logits)
    model.test_targets.append(labels)
    model.auroc_metric(logits, labels)
    model.accuracy_metric(probs, labels)
    model.pr_curve_metric.update(probs, labels)
    model.avg_precision_metric.update(probs, labels)

    mock_logger = MagicMock()
    mock_logger.log_dir = str(tmp_path)

    with patch.object(type(model), "logger", new_callable=PropertyMock, return_value=mock_logger):
        with patch.object(type(model), "current_epoch", new_callable=PropertyMock, return_value=0):
            with patch.object(model, "log"):
                model.on_test_epoch_end()

    # PDF saved
    assert (tmp_path / "pr_curve.pdf").exists()

    # CSV has correct header, one row per threshold point, constant avg_precision
    csv_path = tmp_path / "pr_curve.csv"
    assert csv_path.exists()
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) > 0
    assert set(rows[0].keys()) == {"recall", "precision", "avg_precision"}
    ap_values = {float(row["avg_precision"]) for row in rows}
    assert len(ap_values) == 1
    assert 0.0 <= ap_values.pop() <= 1.0

    # Accumulators and metrics are cleared
    assert model.test_probs == []
    assert model.test_targets == []
    assert model.pr_curve_metric.preds == []
    assert model.avg_precision_metric.preds == []
