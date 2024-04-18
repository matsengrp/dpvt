from ete3 import Tree
from dpvt.generate_data.training_data import (
    good_trees,
    bad_trees,
    mutation_count,
    is_perfect,
    reflect_tree,
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


def test_reflect():
    tree = Tree("(a,(b,c));")
    r_tree = reflect_tree(tree)
    assert r_tree.write(format=9) == "((c,b),a);"
