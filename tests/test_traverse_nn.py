from dpvt.neural_network.models import TraverseNN
from dpvt.generate_data.training_data import good_trees

def test_nn():
    tnn = TraverseNN()
    tree = good_trees[0]
    print(tree.get_ascii(attributes=["sequence", "to_parent"]))
    out = tnn(tree)
    assert out > 0