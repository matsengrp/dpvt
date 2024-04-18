from ete3 import Tree as ete3Tree
# imports for `populate` method
import random
from collections import deque
import itertools


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


def newick_seq_random(tree):
    non_root = tree.write(features=["sequence", "random_tree"], format=9)[:-1]
    root = f"[&&NHX:sequence={tree.sequence}];\n"
    return non_root + root


class Tree(ete3Tree):

    def populate(self, size, names_library=None, reuse_names=False, seed=None,
                 model="fast", ladderize=True, random_branches=False,
                 branch_range=(0,1), support_range=(0,1)):
        """
        Generates a random topology by populating current node.

        :argument size: int, number of leaves to add under current node

        :argument None names_library: If provided, names library
          (list, set, dict, etc.) will be used to name nodes.

        :argument False reuse_names: If True, node names will not be
          necessarily unique, which makes the process a bit more
          efficient.

        :argument None seed: If provided, seed for random number generator

        :argument "fast" distribution: Determines the algorithm used to place
          leaves, which controls the resulting distribution over possible
          topologies. Parameter can be "fast" (original implementation), 
          "yule", or "pda" aka "uniform".
            "fast": newly added leaves are stored in a deque (two-sided linked
              list), in each step a leaf is chosen from one end randomly, and
              the chosen leaf grows two new children.
            "yule": newly added leaves are stored in a list; in each step a
              leaf is chosen randomly from anywhere in the list, and the chosen
              leaf grows two new children. The leaf names are shuffled before
              assigning names to leaves.
            "uniform" or "pda": newly added leaves and interior nodes are
              stored in a list; in each step a node (interior or tip) is chosen
              randomly from anywhere in the list, and the chosen node grows a
              new sister leaf. The leaf names are shuffled before assigning
              names to leaves.

        :argument True ladderize: If True, newly populated subtree is
          ladderized before returning.

        :argument False random_branches: If True, branch distances and support
          values will be randomized.

        :argument (0,1) branch_range: If random_branches is True, this
          range of values will be used to generate random distances.

        :argument (0,1) support_range: If random_branches is True,
          this range of values will be used to generate random branch
          support values.
        """
        NewNode = self.__class__

        if size <= 0:
            return
        if not reuse_names and names_library is not None:
            if size > len(names_library):
                raise ValueError("not enough names provided in names_library")
        random.seed(seed)

        if len(self.children) > 1:
            # add `connector` node between current node `self` and its children
            connector = NewNode()
            for ch in self.get_children():
                ch.detach()
                connector.add_child(child=ch)
            self.add_child(child=connector)
        if len(self.children) > 0:
            # add new `root` under `self` where additional nodes will populate
            # a subtree
            root = self.add_child()
        else:
            root = self

        if model == "fast":
            new_leaves = deque([root])
            for _ in range(size - 1):
                if random.randint(0, 1):
                    p = new_leaves.pop()
                else:
                    p = new_leaves.popleft()

                c1 = p.add_child()
                c2 = p.add_child()
                new_leaves.extend([c1, c2])
                if random_branches:
                    for c in [c1, c2]:
                        c.dist = random.uniform(*branch_range)
                        c.support = random.uniform(*support_range)
                # else: DEFAULT_DIST and DEFAULT_SUPPORT values will be used
        elif model == "yule":
            if size == 1:
                new_leaves = [root]
            elif size >= 2:
                c1 = root.add_child()
                c2 = root.add_child()
                new_leaves = [c1, c2]
            for _ in range(size - 2):
                # choose random leaf
                prev_leaf = random.choice(new_leaves)

                old_parent = prev_leaf.up
                # new internal node below `old_parent`
                new_parent = old_parent.add_child()
                new_leaf = new_parent.add_child()
                prev_leaf.detach()
                new_parent.add_child(child=prev_leaf)
                new_leaves.append(new_leaf)
                c1, c2 = new_leaf, new_parent
                if random_branches:
                    for c in [c1, c2]:
                        c.dist = random.uniform(*branch_range)
                        c.support = random.uniform(*support_range)
        elif model == "pda" or model == "uniform":
            new_leaves = [root]
            new_nodes = [root]
            for _ in range(size - 1):
                # choose random node to add new leaf as sister
                grow_node = random.choice(new_nodes)
                if grow_node != root:
                    # `grow_node` has a parent node
                    old_parent = grow_node.up
                    new_parent = old_parent.add_child()
                    grow_node.detach()
                    new_parent.add_child(child=grow_node)
                    # add child to new_node
                    new_leaf = new_parent.add_child()
                else:
                    # `grow_node` is the root; sister has no parent
                    new_parent = NewNode()
                    if grow_node.is_leaf():
                        new_leaves.remove(grow_node)
                        new_leaves.append(new_parent)
                    for child in grow_node.get_children():
                        child.detach()
                        new_parent.add_child(child=child)
                    grow_node.add_child(child=new_parent)
                    new_leaf = grow_node.add_child()
                
                # add new node, leaf to `new_nodes`, `new_leaves`
                new_leaves.append(new_leaf)
                new_nodes.extend([new_parent, new_leaf])
                if random_branches:
                    for c in [new_parent, new_leaf]:
                        c.dist = random.uniform(*branch_range)
                        c.support = random.uniform(*support_range)
        else:
            raise ValueError(f"parameter topology={model} not recognized")
        if ladderize:
            root.ladderize()

        # assign names to leaf nodes in `new_leaves`
        if names_library is not None:
            names_library = deque(names_library)
        else:
            charset =  "abcdefghijklmnopqrstuvwxyz"
            avail_names = itertools.combinations_with_replacement(charset, 10)
        if model != "fast":
            # shuffle `new_leaves` in random order
            random.shuffle(new_leaves)
        for n in new_leaves:
            if names_library is not None:
                # choose next name
                tname = names_library.popleft()
                if reuse_names:
                    names_library.append(tname)
            else:
                tname = ''.join(next(avail_names))
            n.name = tname