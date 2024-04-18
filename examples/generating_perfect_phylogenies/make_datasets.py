import pickle
import torch
from ete3 import Tree

from dpvt.scripts.utils import Tree as MyTree
from dpvt.scripts.perfect_phylogeny import PerfectPhylogeny
from dpvt.scripts.perturb_phylogeny import (
    perturb_tree,
    sankoff_for_missing_sequences,
)

def create_training_data(file_path, n_trees, n_leaves):
    """
    Create a collection of phylogenies obtained by randomly mixing a subtree of a perfect
    phylogeny.
    """
    DEPTH = 4
    n_phylos_per_tree = 32
    tree_data_dict = {}
    for _ in range(n_trees):
        tree = MyTree()
        tree.populate(n_leaves, model="uniform")
        pp = PerfectPhylogeny(tree)
        for _ in range(n_phylos_per_tree):
            phylo = pp.make_random_phylogeny()
            mixed_phylo = perturb_tree(phylo, depth=DEPTH, exception_on_fail=True)
            # assert(mixed_phylo is not None)
            sankoff_for_missing_sequences(mixed_phylo)
            # convert custom Tree object to ete3 Tree
            newick = mixed_phylo.write(
                features=["sequence", "random_tree"], 
                format_root_node=True
            )
            mixed_phylo = Tree(newick)
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
    # shuffle keys and make train / validation split 
    num_items = n_trees * n_phylos_per_tree
    num_train = int(num_items * 0.8)

    keys = list(tree_data_dict.keys())
    random_idx = torch.randperm(num_items)
    train_keys = [keys[i] for i in random_idx[:num_train]]
    val_keys = [keys[i] for i in random_idx[num_train:]]

    train_data = {key: tree_data_dict[key] for key in train_keys}
    val_data = {key: tree_data_dict[key] for key in val_keys}

    data_dict = {"train": train_data, "val": val_data}
    with open(file_path, "wb") as fh:
        pickle.dump(data_dict, file=fh)

N_LEAVES = 30
def main():
    create_training_data(
        file_path=f"{N_LEAVES}leaf_perfect.p", 
        n_trees=32, 
        n_leaves=N_LEAVES
    )

if __name__ == "__main__":
    main()