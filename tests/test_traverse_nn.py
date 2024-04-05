from dpvt.neural_network.models import TraverseNN
from dpvt.generate_data.training_data import (
    good_trees,
    site4_good_trees,
)


def test_nn():
    tnn = TraverseNN(learning_rate=0.01)
    # tree = good_trees[0]
    tree = site4_good_trees[0]
    print(tree.get_ascii(attributes=["sequence", "to_parent"]))
    out = tnn([tree])
    for x in out:
        assert x.item() > 0
