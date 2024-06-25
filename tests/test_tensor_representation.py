from ete3 import Tree

from dpvt.wrapper import TraversalDataset

import torch

# dictionary for assigning node sequences
node_to_sequence = {
    "s1": "AA",
    "s2": "CA",
    "s3": "CG",
    "s4": "TG",
    "s5": "GG",
    "i1": "CA",
    "i2": "CG",
    "i3": "GG",
    "a1": "GG",
    "a2": "CG",
    "a3": "CG",
    "a4": "CG",
}


def assign_sequences(trees, seq_dict, expect_internal_sequences=True):
    # assign sequences to all nodes in trees
    # seq_dict: dict with node.name : node.sequence for this assignment
    for tree in trees:
        for node in tree.traverse():
            if node.name in seq_dict:
                node.sequence = seq_dict[node.name]
            elif expect_internal_sequences:
                raise ValueError(f"No sequence found for node '{node.name}'")
            else:
                node.sequence = ""
            if node.is_leaf():
                node.node_id = node.name
            else:
                node.node_id = ""
    return trees


def create_test_trees():
    # create two trees for testing
    tree1_nwk = "((((s1,s2)i1,s3)i2,s4)i3,s5)i3;"
    tree2_nwk = "((s1,s2)i1,((s4,s5)i3,s3)i2)i2;"
    tree1 = Tree(tree1_nwk, format=8)
    tree2 = Tree(tree2_nwk, format=8)
    trees = [tree1, tree2]
    assign_sequences(trees, node_to_sequence)
    return trees


def trees_rooted_at_outgroup():
    # Tree identical to those in create_test_trees(), but with outgroup s5
    tree1_nwk = "((((s1,s2)i1,s3)i2,s4)s5)s5;"
    tree2_nwk = "(((s1,s2)i1,(s4,s5)i3)s3)s3;"
    tree1 = Tree(tree1_nwk, format=8)
    tree2 = Tree(tree2_nwk, format=8)
    trees = [tree1, tree2]
    assign_sequences(trees, node_to_sequence)
    return trees


def test_masking():
    trees = trees_rooted_at_outgroup()
    labels = []
    traversal_data = TraversalDataset(trees, labels)
    tree1_masks = [False, False, True, True, False, False, False, False]
    tree2_masks = [False, False, True, False, False, True, False, False]
    excepted_masks = [tree1_masks, tree2_masks]
    print("Expected masks:")
    print(excepted_masks)
    print("Computed masks:")
    print(traversal_data.mask)
    assert traversal_data.mask == excepted_masks


def test_get_mutations():
    trees = trees_rooted_at_outgroup()
    labels = []
    traversal_data = TraversalDataset(trees, labels)
    mutations = traversal_data.mutations
    num_trees = len(trees)
    num_nodes = len(list(trees[0].traverse()))
    num_sites = len(trees[0].sequence)
    expected_mutations = torch.zeros(num_trees, num_nodes, num_sites, 4)
    # first tree
    expected_mutations[0, 2, 0, 1] = -1
    expected_mutations[0, 2, 0, 2] = 1
    expected_mutations[0, 3, 1, 0] = 1
    expected_mutations[0, 3, 1, 1] = -1
    expected_mutations[0, 4, 0, 0] = 1
    expected_mutations[0, 4, 0, 2] = -1
    expected_mutations[0, 7, 0, 1] = -1
    expected_mutations[0, 7, 0, 3] = 1

    # 2nd tree
    expected_mutations[1, 2, 1, 0] = 1
    expected_mutations[1, 2, 1, 1] = -1
    expected_mutations[1, 3, 0, 0] = 1
    expected_mutations[1, 3, 0, 2] = -1
    expected_mutations[1, 5, 0, 1] = 1
    expected_mutations[1, 5, 0, 2] = -1
    expected_mutations[1, 6, 0, 1] = -1
    expected_mutations[1, 6, 0, 3] = 1
    assert torch.equal(expected_mutations, mutations)


def test_get_traversal():
    trees = trees_rooted_at_outgroup()
    labels = []
    traversal_data = TraversalDataset(trees, labels)
    traversal = traversal_data.traversal
    num_trees = len(trees)
    num_int_nodes = len(trees[0]) - 2
    expected_traversal = torch.zeros(num_trees, 2, num_int_nodes, 3)
    # first tree
    expected_traversal[0, 0, 0, :] = torch.tensor([4, 5, 3])
    expected_traversal[0, 0, 1, :] = torch.tensor([3, 6, 2])
    expected_traversal[0, 1, 0, :] = torch.tensor([7, 1, 2])
    expected_traversal[0, 1, 1, :] = torch.tensor([6, 2, 3])

    # 2nd tree
    expected_traversal[1, 0, 0, :] = torch.tensor([3, 4, 2])
    expected_traversal[1, 0, 1, :] = torch.tensor([6, 7, 5])
    expected_traversal[1, 1, 0, :] = torch.tensor([5, 1, 2])
    expected_traversal[1, 1, 1, :] = torch.tensor([2, 1, 5])

    print("Expected:")
    print(expected_traversal)
    print("Computed:")
    print(traversal)
    assert torch.equal(expected_traversal, traversal)
