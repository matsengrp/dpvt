#!/usr/bin/env python
import click
from ete3 import Tree
from perfect_phylogeny import perfect_phylogeny


h1 = "Label the nodes of the topology with sequences."
h2 = "Label the nodes of the topology with substitutions."
h3 = "Require leaf nodes to have unique sequences."
h4 = "Require no two site columns are identical (among all nodes)."
h5 = "Require every edge has at least one substitution."
h6 = "Require every non-terminal edge has at least one substitution."
h7 = "Maximum number of sites in the sequence alignment."


@click.command()
@click.argument("input_file")
@click.argument("output_file")
@click.option("--seq/--no-seq", "use_seq", default=True, help=h1)
@click.option("--sub/--no-sub", "use_sub", default=True, help=h2)
@click.option("--unique-leaves", is_flag=False, flag_value=True, default=False, help=h3)
@click.option(
    "--distinct-sites", is_flag=False, flag_value=True, default=False, help=h4
)
@click.option(
    "--sub-on-all-edges", is_flag=False, flag_value=True, default=False, help=h5
)
@click.option(
    "--sub-on-all-internal", is_flag=False, flag_value=True, default=False, help=h6
)
@click.option("--max-sites", default=1, help=h7)
def run(
    input_file,
    output_file,
    use_seq=True,
    use_sub=True,
    unique_leaves=False,
    distinct_sites=False,
    sub_on_all_edges=False,
    sub_on_all_internal=False,
    max_sites=1,
):
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
                phylogeny_maker = perfect_phylogeny(tree)
                p_phylos = phylogeny_maker.make_trees(
                    use_seq,
                    use_sub,
                    unique_leaves,
                    distinct_sites,
                    sub_on_all_edges,
                    sub_on_all_internal,
                    max_sites,
                )
                j = -1
                for j, p_phylo in enumerate(p_phylos):
                    out_file.write(newick_format(p_phylo))
                print(f"Wrote {j+1} perfect phylogenies for topology {i}.")


def newick_bare(tree):
    return tree.write(format=9)


def newick_seq(tree):
    non_root = tree.write(features=["sequence"], format=9)[:-1]
    root = f"[&&NHX:sequence={tree.sequence}];\n"
    return non_root + root


def newick_sub(tree):
    non_root = tree.write(features=["sub"], format=9)[:-1]
    root = "[&&NHX:sub={}];\n"
    return non_root + root


def newick_seq_sub(tree):
    non_root = tree.write(features=["sequence", "sub"], format=9)[:-1]
    root = f"[&&NHX:sequence={tree.sequence}:sub=" + "{}];\n"
    return non_root + root


if __name__ == "__main__":
    run()
