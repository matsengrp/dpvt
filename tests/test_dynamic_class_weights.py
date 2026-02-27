import torch

from dpvt.models import TraverseNN


def test_unweighted_loss_unchanged():
    """With dynamic_class_weights=False, loss matches the default behavior."""
    model = TraverseNN(dynamic_class_weights=False)

    pred = torch.randn(2, 5, 1)
    target = torch.tensor([[0, 1, 0, 1, 0], [1, 0, 0, 1, 0]], dtype=torch.float32)
    mask = torch.ones(2, 5, dtype=torch.bool)

    loss = model.masked_bce_loss(pred, target, mask)

    # Compute expected loss manually (no reweighting)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        pred, target.unsqueeze(-1), reduction="mean"
    )
    assert torch.allclose(loss, expected, atol=1e-6)


def test_weighted_loss_differs_on_imbalanced_data():
    """With dynamic_class_weights=True, imbalanced labels produce different loss."""
    torch.manual_seed(42)

    pred = torch.randn(2, 10, 1)
    # Heavily imbalanced: mostly 0s with few 1s
    target = torch.tensor(
        [[0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]],
        dtype=torch.float32,
    )
    mask = torch.ones(2, 10, dtype=torch.bool)

    model_unweighted = TraverseNN(dynamic_class_weights=False)
    model_weighted = TraverseNN(dynamic_class_weights=True)

    loss_unweighted = model_unweighted.masked_bce_loss(pred, target, mask)
    loss_weighted = model_weighted.masked_bce_loss(pred, target, mask)

    assert not torch.allclose(loss_unweighted, loss_weighted, atol=1e-6), (
        "Weighted and unweighted loss should differ on imbalanced data"
    )


def test_weighted_loss_equals_unweighted_on_balanced_data():
    """When classes are perfectly balanced, weighting should not change the loss."""
    torch.manual_seed(0)

    pred = torch.randn(2, 4, 1)
    # Perfectly balanced: 2 positives and 2 negatives per sample
    target = torch.tensor(
        [[0, 1, 0, 1], [1, 0, 1, 0]],
        dtype=torch.float32,
    )
    mask = torch.ones(2, 4, dtype=torch.bool)

    model_unweighted = TraverseNN(dynamic_class_weights=False)
    model_weighted = TraverseNN(dynamic_class_weights=True)

    loss_unweighted = model_unweighted.masked_bce_loss(pred, target, mask)
    loss_weighted = model_weighted.masked_bce_loss(pred, target, mask)

    assert torch.allclose(loss_unweighted, loss_weighted, atol=1e-6), (
        "Weighted and unweighted loss should be equal on balanced data"
    )


def test_weighted_loss_respects_mask():
    """Masked-out edges should not contribute to the loss or weight calculation."""
    torch.manual_seed(7)

    pred = torch.randn(1, 6, 1)
    target = torch.tensor([[0, 0, 0, 0, 1, 0]], dtype=torch.float32)
    mask_full = torch.ones(1, 6, dtype=torch.bool)
    mask_partial = torch.tensor([[True, True, False, False, True, False]])

    model = TraverseNN(dynamic_class_weights=True)

    loss_full = model.masked_bce_loss(pred, target, mask_full)
    loss_partial = model.masked_bce_loss(pred, target, mask_partial)

    # Losses should differ because different edges are included
    assert not torch.allclose(loss_full, loss_partial, atol=1e-6)
