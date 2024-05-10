#!/usr/bin/env python
import click
from ete3 import Tree
from dpvt.scripts.perfect_phylogeny import PerfectPhylogeny
from dpvt.scripts.utils import newick_bare, newick_seq, newick_sub, newick_seq_sub


h1 = "Label the nodes of the topology with sequences."
h2 = "Label the nodes of the topology with substitutions."
h3 = "Require leaf nodes to have unique sequences."
h4 = "Require every edge has at least one substitution."
h5 = "Minimum number of sites in the sequence alignment."
h6 = "Maximum number of sites in the sequence alignment."


@click.command()
@click.argument("input_file")
@click.argument("output_file")
@click.option("--seq/--no-seq", "use_seq", default=True, help=h1)
@click.option("--sub/--no-sub", "use_sub", default=True, help=h2)
@click.option("--unique-leaves", is_flag=False, flag_value=True, default=False, help=h3)
@click.option("--sub-all-edges", is_flag=False, flag_value=True, default=False, help=h4)
@click.option("--min-sites", default=1, help=h5)
@click.option("--max-sites", default=None, type=int, help=h6)
def run(
    input_file,
    output_file,
    use_seq=True,
    use_sub=True,
    unique_leaves=False,
    sub_all_edges=False,
    min_sites=1,
    max_sites=None,
):
    """
    Args:
        input_file: contains newick strings for unlabelled topologies, one on each line
        output_file: file to write perfect phylogenies, in extended newick format, one
            per line
    """
    newick_format = {
        (False, False): newick_bare,
        (False, True): newick_sub,
        (True, False): newick_seq,
        (True, True): newick_seq_sub,
    }[(use_seq, use_sub)]

    with open(input_file) as in_file:
        with open(output_file, "w") as out_file:
            for i, newick in enumerate(in_file):
                print(f"Processing topology {i}.")
                tree = Tree(newick)
                phylogeny_maker = PerfectPhylogeny(tree)
                p_phylos = phylogeny_maker.make_phylogenies(
                    use_seq, use_sub, unique_leaves, sub_all_edges, min_sites, max_sites
                )
                j = -1
                for j, p_phylo in enumerate(p_phylos):
                    out_file.write(newick_format(p_phylo))
                    if (j + 1) % 10000 == 0:
                        print(f"  {j+1} phylogenies found...")
                print(f"Wrote {j+1} perfect phylogenies for topology {i}.")


if __name__ == "__main__":
    run()
