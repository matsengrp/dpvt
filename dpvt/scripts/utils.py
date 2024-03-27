from ete3 import Tree


def newick_bare(tree):
    return tree.write(format=9)


def newick_seq(tree):
    non_root = tree.write(features=["sequence"], format=9)[:-1]
    root = f"[&&NHX:sequence={tree.sequence}];\n"
    return non_root + root


def newick_sub(tree):
    non_root = tree.write(features=["subs"], format=9)[:-1]
    root = "[&&NHX:subs={}];\n"
    return non_root + root


def newick_seq_sub(tree):
    non_root = tree.write(features=["sequence", "subs"], format=9)[:-1]
    root = f"[&&NHX:sequence={tree.sequence}:subs={{}}];\n"
    return non_root + root
