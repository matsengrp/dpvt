from dpvt.neural_network.traverse_nn import TraverseNN
from dpvt.neural_network.training_data import (
    good_trees,
    site4_good_trees,
)

def test_nn():
    tnn = TraverseNN()
    # tree = good_trees[0]
    tree = site4_good_trees[0]
    print(tree.get_ascii(attributes=["sequence", "to_parent"]))
    out = tnn([tree])
    assert out > 0