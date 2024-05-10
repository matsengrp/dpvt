from ete3 import Tree
from historydag.parsimony import parsimony_score
from dpvt.generate_data.perfect_phylogeny import PerfectPhylogeny
from dpvt.generate_data.perturb_phylogeny import make_worse_tree
from dpvt.generate_data.utils import newick_seq, newick_seq_random, populate
import pandas as pd
import numpy as np

# A basic example of generating random topologies, assigning a perfect phylogeny,
# and perturbing to get a higher parsimony score.


def try_it():
    tree_count = 10
    max_attempts = 100
    results = {
        "leaf_count": [],
        "initial_tree": [],
        "initial_score": [],
        "perturbed_tree": [],
        "perturbed_score": [],
        "perturbed_depth": [],
    }

    for leaf_count in [5, 10, 15, 20, 25]:
        for _ in range(tree_count):
            tree = Tree()
            populate(tree, leaf_count, model="uniform")
            phylogeny_maker = PerfectPhylogeny(tree)
            tree = phylogeny_maker.make_random_phylogeny(
                use_seq=True,
                use_sub=False,
                unique_leaves=False,
                sub_on_all_edges=False,
            )
            for depth in [3, 4, 5]:
                results["leaf_count"].append(leaf_count)
                results["initial_tree"].append(newick_seq(tree))
                results["initial_score"].append(parsimony_score(tree))
                results["perturbed_depth"].append(depth)
                p_tree = make_worse_tree(tree, depth, max_attempts=max_attempts)
                if p_tree is not None:
                    results["perturbed_tree"].append(newick_seq_random(p_tree))
                    results["perturbed_score"].append(parsimony_score(p_tree))
                else:
                    results["perturbed_tree"].append("")
                    results["perturbed_score"].append(np.nan)

    pd.DataFrame(results).to_csv("perturbed_trees.csv")


if __name__ == "__main__":
    try_it()
