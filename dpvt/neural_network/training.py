from ete3 import Tree

# tree with maximum parsimony
t_good = Tree("(0,(1,2));")
for node in (t_good, t_good.children[0]):
    node.sequence = "A"
for node in (t_good.children[1], t_good.children[1].children[0], t_good.children[1].children[1]):
    node.sequence = "T"
print(t_good.get_ascii(attributes=["sequence"]))

# tree without maximum parsimony
t_bad = Tree("(0,(1,2));")
for node in (t_bad, t_bad.children[1], t_bad.children[1].children[1]):
    node.sequence = "A"
for node in (t_bad.children[0], t_bad.children[1].children[0]):
    node.sequence = "T"
print(t_bad.get_ascii(attributes=["sequence"]))
