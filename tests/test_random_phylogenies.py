# from ete3 import Tree
# from ete4 import Tree as ete4Tree
from dpvt.scripts.perturb_phylogeny import Tree, perturb_tree
from dpvt.scripts.perfect_phylogeny import PerfectPhylogeny
from dpvt.scripts.utils import newick_sub
import time
import matplotlib.pyplot as plt
import pandas as pd

def test_make_random_phylo():
    t = Tree()
    t.populate(10, model="uniform")
    pp = PerfectPhylogeny(t)
    phylo = pp.make_random_phylogeny()
    assert(len(phylo.get_leaf_names()) == 10)

def test_random_perturb():
    t = Tree()
    t.populate(10, model="uniform")
    pp = PerfectPhylogeny(t)
    phylo = pp.make_random_phylogeny()
    phylo = perturb_tree(phylo, depth=3)
    assert(len(phylo.get_leaf_names()) == 10)

def check_randomness():
    # Currently not uniform random. There are hotspots. It seems worse when requiring
    # substitutions on all internal edges instead of all edges. Is that because there are
    # more of the latter or is there something about the enumeration process?
    reps = 1000

    # We can't go much further with the leaf count, because the number of perfect
    # phylogenies blows up so fast.
    for leaf_count in range(3, 6):
        tree = Tree()
        tree.populate(leaf_count, model="uniform")
        tree = Tree(tree.write())
        phylogeny_maker = PerfectPhylogeny(tree)
        for all_edges in [False, True]:
            perfect_phylogenies = {
                newick_sub(phylo): 0
                for phylo in phylogeny_maker.make_phylogenies(
                    use_seq=False,
                    use_sub=True,
                    unique_leaves=False,
                    sub_on_all_edges=all_edges,
                    skip_perms=True,
                    shuffle=False,
                )
            }
            for _ in range(reps):
                phylo = phylogeny_maker.make_random_phylogeny(
                    use_seq=False,
                    use_sub=True,
                    unique_leaves=False,
                    sub_on_all_edges=all_edges,
                )
                perfect_phylogenies[newick_sub(phylo)] += 1

            fig, ax = plt.subplots()
            xs, ys = zip(*enumerate((p for p in perfect_phylogenies.values() if p > 0)))
            ax.scatter(xs, ys)
            ax.set_ylabel("sample count")
            ax.set_xlabel("index")
            ax.set_ylim(bottom=0, top=max(ys) + 1)
            total = len(perfect_phylogenies)
            missed = total - len(xs)
            ax.set_title(
                f"Number of times distinct perfect phylogenies are sampled.\nOf the "
                + f"{total} total, {missed} were never sampled."
            )
            edges = "all" if all_edges else "internal"
            plt.savefig(f"random_{leaf_count}_leaves_{edges}_edges.png")
            plt.clf()


def check_runtime():
    """
    Probably cubic-ish. This will run in about 5 minutes.
    """
    results = {
        "leaf_count": [],
        "random_topology_time": [],
        "perfect_phylogeny_init_time": [],
        "generate_random_time": [],
        "generate_random_all_edges_time": [],
    }
    derp_time = -time.time()
    reps = 10
    for leaf_count in range(5, 65, 5):
        results["leaf_count"].append(leaf_count)
        trees = []
        p_phylo_makers = []

        runtime = -time.time()
        for _ in range(reps):
            tree = Tree()
            tree.populate(leaf_count, model="uniform")
            tree = Tree(tree.write())
            trees.append(tree)
        runtime += time.time()
        results["random_topology_time"].append(runtime / reps)

        runtime = -time.time()
        for tree in trees:
            phylogeny_maker = PerfectPhylogeny(tree)
            p_phylo_makers.append(phylogeny_maker)
        runtime += time.time()
        results["perfect_phylogeny_init_time"].append(runtime / reps)

        runtime = -time.time()
        for phylogeny_maker in p_phylo_makers:
            p_phylo = phylogeny_maker.make_random_phylogeny(
                use_seq=True,
                use_sub=False,
                unique_leaves=False,
                sub_on_all_edges=False,
            )
        runtime += time.time()
        results["generate_random_time"].append(runtime / reps)

        runtime = -time.time()
        for phylogeny_maker in p_phylo_makers:
            p_phylo = phylogeny_maker.make_random_phylogeny(
                use_seq=True,
                use_sub=False,
                unique_leaves=False,
                sub_on_all_edges=True,
            )
        runtime += time.time()
        results["generate_random_all_edges_time"].append(runtime / reps)

    pd.DataFrame(results).to_csv("runtime.csv")


if __name__ == "__main__":
    check_randomness()
    check_runtime()
