from dpvt.neural_network.models import (
    TraverseNN,
    TransformerEncoderTraversal,
)
from dpvt.generate_data.training_data import (
    good_trees,
    site4_good_trees,
)


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


def test_transformer_first_nn():
    model = TransformerEncoderTraversal()

    tree = good_trees[0]
    out = model([tree])
    for x in out[0]:
        assert x.item() > 0

    tree = site4_good_trees[0]
    out = model([tree])
    for x in out[0]:
        assert x.item() > 0
