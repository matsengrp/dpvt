import pickle
import torch
from ete3 import Tree

from dpvt.generate_data.utils import populate
from dpvt.generate_data.perfect_phylogeny import PerfectPhylogeny
from dpvt.generate_data.perturb_phylogeny import (
    perturb_tree,
    sankoff_for_missing_sequences,
)

def create_training_data(n_trees, n_phylos_per_tree, n_leaves, tree_depth):
    """
    Create a collection of phylogenies obtained by randomly mixing a subtree of a
    perfect phylogeny.
    """
    tree_data_dict = {}
    for _ in range(n_trees):
        # create a random tree
        tree = Tree()
        populate(tree, n_leaves, model="uniform")
        pp = PerfectPhylogeny(tree)
        for _ in range(n_phylos_per_tree):
            # create a random perfect phylogeny
            phylo = pp.make_random_phylogeny()
            mixed_phylo = perturb_tree(phylo, depth=tree_depth, exception_on_fail=True)
            # assert(mixed_phylo is not None)
            # generate sequences on internal nodes
            sankoff_for_missing_sequences(mixed_phylo)
            # add "extra" unifurcating root above previous root
            new_tree = Tree()
            new_tree.add_child(mixed_phylo)
            new_tree.sequence = mixed_phylo.sequence
            new_tree.random_tree = mixed_phylo.random_tree
            mixed_phylo = new_tree
            edge_classifier = [
                1.0 if (node.random_tree and not node.is_leaf()) else 0.0
                for node in mixed_phylo.traverse(strategy="preorder")
            ]
            tree_data_dict[mixed_phylo] = edge_classifier
    return tree_data_dict

N_LEAVES = 30
DEPTH = 4
def main():
    data_dict = create_training_data(
        n_trees=32, 
        n_phylos_per_tree=32,
        n_leaves=N_LEAVES,
        tree_depth=DEPTH,
    )
    file_path=f"{N_LEAVES}leaf_perfect.p"
    with open(file_path, "wb") as fh:
        pickle.dump(data_dict, file=fh)

if __name__ == "__main__":
    main()