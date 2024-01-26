from dpvt.scripts.perfect_phylogeny import (
    PerfectPhylogeny
)
from ete3 import Tree

def test_3_leaf_1_site():
    t = Tree("((0,1),2);")
    pp = PerfectPhylogeny(t)
    tree_gen = pp.make_trees(
        use_seq=True,
        use_sub=False,
        unique_leaves=True,
        sub_on_all_internal=True,
        max_sites=1
    )
    tree_list = list(tree_gen)
    assert len(tree_list) == 120
    expected_newick = (
        "((0[&&NHX:sequence=C],1[&&NHX:sequence=G])[&&NHX:sequence=G],2[&&NHX:sequence="
        "A]);"
    )
    assert tree_list[0].write(format=9, features=["sequence"]) == expected_newick

def test_5_leaf_2_sites():
    t = Tree("(0,(1,((2,3),4)));")
    pp = PerfectPhylogeny(t)
    tree_gen = pp.make_trees(
        use_seq=False,
        use_sub=True,
        unique_leaves=True,
        sub_on_all_internal=True,
        max_sites=2
    )
    expected_newick = (
        "(0[&&NHX:subs={}],(1[&&NHX:subs={}],((2[&&NHX:subs={C1T}],3[&&NHX:subs={}])[&&"
        "NHX:subs={G1C}],4[&&NHX:subs={}])[&&NHX:subs={A1G}])[&&NHX:subs={A0G}]);"
    )
    assert next(tree_gen).write(format=9, features=["subs"]) == expected_newick
    # tree_list = list(tree_gen)
    # assert len(tree_list) == 179136
