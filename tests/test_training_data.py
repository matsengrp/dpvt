from dpvt.neural_network.training_data import (
    good_trees,
    bad_trees,
    mutation_count,
    is_perfect,
)


def test_mutation_count():
    for tree in good_trees:
        assert mutation_count(tree) == 1
    for tree in bad_trees:
        assert mutation_count(tree) == 2


def test_is_perfect():
    for tree in good_trees:
        assert is_perfect(tree)
    for tree in bad_trees:
        assert not is_perfect(tree)
