from ete3 import Tree
from dpvt.generate_data.utils import newick_seq, populate
from dpvt.generate_data.random_perfect_phylogeny import RandomPerfectPhylogeny
from dpvt.generate_data.perfect_phylogeny import PerfectPhylogeny
from time import time
from itertools import permutations, product
import re
import pandas as pd


# These scripts check that the perfect phylogenies from RandomPerfectPhylogeny agrees
# with those from PerfectPhylogeny, and times RandomPerfectPhylogeny. The runtime is
# non-linear in the number of leaves (cubic maybe?), with the blowup beginning around
# 100 leaves. E.g., generating a random perfect phylogeny on 100 leaves takes about 0.02
# seconds, 500 leaves takes about 2.2 seconds, and 1000 leaves takes 17.2 seconds.


def expand_by_site_permutations(newick_string):
    """
    Take in a newick string, with the sequence attribute at nodes and no other node
    features, and returns the (frozen) set of the newick strings given by permuting the
    site order of sequences.
    """
    sequence_length = newick_string[newick_string.index("=") + 1 :].index("]")
    re_pattern = r"sequence=.{" + str(sequence_length) + "}"
    regex = re.compile(re_pattern)
    newicks = frozenset(
        (
            regex.sub(lambda m: re_replacement1(m, perm), newick_string)
            for perm in permutations(range(sequence_length))
        )
    )
    return newicks


def expand_by_site_and_base_permutations(newick_string):
    """
    Take in a newick string, with the sequence attribute at nodes and no other node
    features, and returns a generator for the newick strings given by permuting the
    site order of sequences and permuting the bases at each site.
    """
    sequence_length = newick_string[newick_string.index("=") + 1 :].index("]")
    re_pattern = r"sequence=.{" + str(sequence_length) + "}"
    regex = re.compile(re_pattern)
    bases = "AGCT"
    all_base_perms = [dict(zip(bases, perm)) for perm in permutations(bases)]
    return (
        regex.sub(lambda m: re_replacement2(m, s_perm, b_perms), newick_string)
        for s_perm in permutations(range(sequence_length))
        for b_perms in product(all_base_perms, repeat=sequence_length)
    )


def re_replacement1(matchobj, perm):
    the_string = matchobj.group(0)
    old_seq = the_string[9:]
    new_seq = "".join(old_seq[i] for i in perm)
    return the_string[:9] + new_seq


def re_replacement2(matchobj, site_perm, base_perms):
    the_string = matchobj.group(0)
    old_seq = the_string[9:]
    new_seq = "".join(base_perms[i][old_seq[i]] for i in site_perm)
    return the_string[:9] + new_seq


def check_all_found(nwk):
    """
    Checks if the new method of generating random perfect phylogenies finds the same
    perfect phylogenies as the old method of generating all random perfect phylogenies,
    up to permutation of site order. This does not additionally check for permutations
    of bases. This method is too slow for topologies with 6 or more leaves.
    """
    tree = Tree(nwk)
    pp = PerfectPhylogeny(tree)
    rpp = RandomPerfectPhylogeny(tree, edges="internal", no_permutations=True)

    all_old_perfect_phylogenies = pp.make_phylogenies(
        use_seq=True,
        use_sub=False,
        unique_leaves=False,
        sub_on_all_edges=False,
        min_sites=1,
        max_sites=None,
        skip_perms=True,
        shuffle=False,
    )
    old_perfect_nwks = map(newick_seq, all_old_perfect_phylogenies)
    old_perfect_expanded_nwks = set(map(expand_by_site_permutations, old_perfect_nwks))

    sample_count = 100 * len(old_perfect_expanded_nwks)
    many_new_random = (
        rpp.make_random_perfect_phylogeny(use_seq=True, use_sub=False)
        for _ in range(sample_count)
    )
    new_random_nwks = map(expand_by_site_permutations, map(newick_seq, many_new_random))

    return old_perfect_expanded_nwks == set(new_random_nwks)


def check_all_valid(nwk, sample_count=500):
    """
    Checks that a random sample of perfect phylogenies from the new method are found by
    the old generator method, after allowing for permutating site order and bases. The
    runtime scales with both the number of leaves and the number of samples.
    """
    tree = Tree(nwk)
    pp = PerfectPhylogeny(tree)
    rpp = RandomPerfectPhylogeny(tree, edges="internal", no_permutations=True)

    all_old_perfect_phylogenies = pp.make_phylogenies(
        use_seq=True,
        use_sub=False,
        unique_leaves=False,
        sub_on_all_edges=False,
        min_sites=1,
        max_sites=None,
        skip_perms=True,
        shuffle=False,
    )
    old_perfect_nwks = set(map(newick_seq, all_old_perfect_phylogenies))

    many_new_random = (
        rpp.make_random_perfect_phylogeny(use_seq=True, use_sub=False)
        for _ in range(sample_count)
    )

    printProgressBar(0, sample_count)
    for i, nwk in enumerate(map(newick_seq, many_new_random)):
        valid = any(
            (
                related_nwk in old_perfect_nwks
                for related_nwk in expand_by_site_and_base_permutations(nwk)
            )
        )
        if not valid:
            print(f"Problem with: {nwk}")
            return False
        printProgressBar(i + 1, sample_count)
    return True


def printProgressBar(iteration, total):
    """
    Progress bar for command line display, taken and modified from stack overflow.
    """
    prefix = "Progress:"
    suffix = "Complete"
    decimals = 1
    length = 50
    fill = "█"
    printEnd = "\r"

    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + "-" * (length - filledLength)
    print(f"\r{prefix} |{bar}| {percent}% {suffix}", end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def check_speed(nwk, reps):
    tree = Tree(nwk)
    runtime = -time()
    rpp = RandomPerfectPhylogeny(tree, edges="internal", no_permutations=True)
    for _ in range(reps):
        temp = rpp.make_random_perfect_phylogeny(use_seq=True, use_sub=False)
    runtime += time()
    return runtime


def get_topologies(leaf_count):
    counts = {
        1: 1,
        2: 1,
        3: 1,
        4: 2,
        5: 3,
        6: 6,
        7: 11,
        8: 23,
        9: 47,
        10: 106,
    }
    # Number of unlabelled topologies. Literally the first example on OEIS.
    sample_count = 100 * counts[leaf_count] if leaf_count in counts else 5000

    newicks = set()
    for _ in range(sample_count):
        tree = Tree()
        populate(tree, leaf_count, model="uniform")
        newicks.add(tree.write(format=100))
    topologies = set()
    for newick in newicks:
        tree = Tree(newick, format=100)
        i = 0
        for leaf in tree.get_leaves():
            leaf.name = i
            i += 1
        topologies.add(tree)
    return topologies


if __name__ == "__main__":

    print("Checking runtime.")
    timing_results = []
    topology_reps = 5
    perfect_phylogeny_reps = 50
    for n in (*range(3, 50), *range(50, 1001, 50)):
        for _ in range(topology_reps):
            tree = Tree()
            populate(tree, n, model="uniform")
            nwk = tree.write(format=9)
            runtime = check_speed(nwk, perfect_phylogeny_reps)
            timing_results.append((n, runtime, runtime / perfect_phylogeny_reps))
    pd.DataFrame(
        timing_results, columns=("leaf_count", "time_to_100", "avg_time")
    ).to_csv("runtime.csv")
    print("Done checking runtime.")

    for n in range(3, 6):
        topologies = get_topologies(n)
        print(f"Checking {len(topologies)} topologies with {n} leaves...")
        found_all = True
        for i, topology in enumerate(topologies, 1):
            print(f"Checking topology {i}...")
            nwk = topology.write(format=9)
            found_all &= check_all_found(nwk)
        print(f"All old perfect phylogies found by new method: {found_all}")

    for n in [6]:  # 7 leaves takes too long
        topologies = get_topologies(n)
        print(f"Checking {len(topologies)} topologies with {n} leaves...")
        all_valid = True
        for i, topology in enumerate(topologies, 1):
            print(f"Checking topology {i}...")
            nwk = topology.write(format=9)
            valid = check_all_valid(nwk, 100)
            all_valid &= valid
            if not valid:
                print(f"Invalid perfect phylogeny from input topology: {nwk}")
        print(f"All new perfect phylogies found by old method: {all_valid}\n")
