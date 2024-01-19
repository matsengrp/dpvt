from ete3 import Tree
from itertools import (
    permutations as perms,
    combinations as combs,
    combinations_with_replacement as combs_r,
    product as prod,
)


class perfect_phylogeny:
    """
    ...descriptions not up to date...

    A class to determine multiple sequence alignments and mutation histories that form
    a perfect phylogeny for a given topology.

    Attributes:
        leaf_indices (list): A list of leaf node indices in the list of nodes.
        leaf_count (int): The number of leaf nodes in the topology.
        mutation_node_index_lists (list): A list of lists, where each inner list
            gives the indices of nodes where mutations occur, such that the mutations
            can be chosen to produce a perfect phylogeny
        nodes (list): A preorder list of the nodes of the topology. The node indices
            used throughout this class are indices into this list.
            ...preorder, so root is at index 0...
        node_count (int): The nunber of nodes in the topology.
        node_index (dict): A dictionary mapping a node of the topology to its index in
            the list of nodes.
        state_count (int): The number of states.
        state_lists: each element is a list that marks a node as 0,1,2,...,num states -1
            where that labelleing will givre a perfect phylo
            after mapping 0,1,2,....num states-1
            to some permutation of the allowed state letters.
            indexed in same order as mutation_node_index_lists
        state_permutation (dict): ...
        states (list): The allowed characters in sequences.
        tree (ete3.Tree): The topology.
    """

    def __init__(self, tree, states=["A", "G", "C", "T"]):
        self.tree = tree.copy()
        self.states = states
        self.state_count = len(self.states)
        self.nodes = list(self.tree.traverse(strategy="preorder"))
        self.node_count = len(self.nodes)
        self.node_index = dict(zip(self.nodes, range(self.node_count)))
        self.leaf_indices = [self.node_index[leaf] for leaf in self.tree.get_leaves()]
        self.leaf_count = len(self.leaf_indices)
        self.state_permutations = {
            r: list(perms(self.states, r)) for r in range(1, self.state_count + 1)
        }

        for node in self.nodes:
            node.add_feature("node_index", self.node_index[node])
        self.make_mutation_index_lists()
        self.make_state_lists()

    def make_mutation_index_lists(self):
        """Initialize self.mutation_node_index_lists."""
        self.mutation_node_index_lists = [
            mutated_node_indices
            for num_subs in range(0, self.state_count)
            for mutated_node_indices in combs(range(1, self.node_count), num_subs)
        ]
        return None

    def make_state_lists(self):
        """Initialize self.state_lists."""
        self.state_lists = list(
            map(self.make_state_list, self.mutation_node_index_lists)
        )
        return None

    def make_state_list(self, node_indices):
        """
        Make an entry for self.state_lists from one of self.mutation_node_index_lists.
        """
        state_list = [0] * self.node_count
        for i, j in enumerate(node_indices, 1):
            state_list[j] = i
        for node, index in self.node_index.items():
            if index not in node_indices and not node.is_root():
                parent_index = self.node_index[node.up]
                state_list[index] = state_list[parent_index]
        return state_list

    def node_sequence(self, node_index, state_lists_indices, perm_indices):
        """
        Returns the sequence for the node at the given index, based on the specified
        state lists and permutations. Specifically, the character at site i of the
        sequence is obtained by applying the j-th permutation (of appropriate size) to
        the entry of the k-th state list corresponding to the node, where j is the i-th
        entry of perm_indices and k is the i-th entry of state_lists_indices.
        """
        r = lambda i: len(self.mutation_node_index_lists[i]) + 1
        perm_fn = lambda r, i, x: self.state_permutations[r][i][x]
        return "".join(
            (
                perm_fn(r(k), j, self.state_lists[k][node_index])
                for j, k in zip(perm_indices, state_lists_indices)
            )
        )

    def node_subs(self, node_index, state_lists_indices, perm_indices):
        """
        Returns the substitutions for the parent of the specified to the node.
        This follows the same format as the node_sequence method.
        """
        # need to work out nicer format for list of subs, but ete is picky about allowed characters in newicks.
        r = lambda i: len(self.mutation_node_index_lists[i]) + 1
        perm_fn = lambda r, i, x: self.state_permutations[r][i][x]
        subs = []
        if node_index != 0:
            parent_index = self.nodes[node_index].up.node_index
            zipped = zip(state_lists_indices, perm_indices)
            for i, (sl_index, perm_index) in enumerate(zipped):
                parent_site = self.state_lists[sl_index][parent_index]
                node_site = self.state_lists[sl_index][node_index]
                if parent_site != node_site:
                    parent = perm_fn(r(sl_index), perm_index, parent_site)
                    child = perm_fn(r(sl_index), perm_index, node_site)
                    subs.append(f"{i}_{parent}->{child}")
        return "{" + "|".join(subs) + "}"

    def make_tree(
        self,
        label_seq,
        label_sub,
        state_lists_indices,
        perm_indices,
        unique_leaves=False,
    ):
        """..."""
        seq = lambda n: self.node_sequence(n, state_lists_indices, perm_indices)
        sub = lambda n: self.node_subs(n, state_lists_indices, perm_indices)

        tree = self.tree.copy()
        for node in tree.traverse(strategy="preorder"):
            node_index = node.node_index
            if label_seq:
                node.add_feature("sequence", seq(node_index))
            if label_sub:
                node.add_feature("sub", sub(node_index))

        is_valid = True
        if unique_leaves:
            if label_seq:
                leaf_seqs = {leaf.sequence for leaf in tree.get_leaves()}
            else:
                leaf_seqs = {seq(leaf.node_index) for leaf in tree.get_leaves()}
            is_valid &= len(leaf_seqs) == self.leaf_count

        return (tree, is_valid)

    def perms_for_states(self, state_list_indices):
        """..."""
        r = lambda i: len(self.mutation_node_index_lists[i]) + 1
        return [
            list(range(len(self.state_permutations[r(i)]))) for i in state_list_indices
        ]

    def make_trees(
        self,
        use_seq=True,
        use_sub=True,
        unique_leaves=False,
        distinct_sites=False,
        max_sites=1,
    ):
        """...only one tree per msa on all nodes (not just leaves), with msa's the same after site rearrangement"""

        reps = perfect_phylogeny.paired_repeat
        next_tree = self.make_tree
        n = len(self.state_lists)
        trees = (
            t[0]
            for m in range(1, max_sites + 1)
            for state_lists in combs_r(range(n), m)
            for perms in prod(*self.perms_for_states(state_lists))
            if not (distinct_sites and reps(state_lists, perms))
            if (t := next_tree(use_seq, use_sub, state_lists, perms, unique_leaves))[1]
        )
        return trees

    @staticmethod
    def paired_repeat(list1, list2):
        """..."""
        n = len(list1) - 1
        return any(
            (list1[i] == list1[i + 1] and list2[i] == list2[i + 1] for i in range(n))
        )

    @staticmethod
    def print_columns(tree):
        """..."""
        if not hasattr(tree, "sequence"):
            raise ValueError("The input tree does not have the sequence attribute.")
        cols = list(zip(*(n.sequence for n in tree.traverse(strategy="preorder"))))
        for i, c in enumerate(cols):
            print(f"column {i}: {c}")
        return None
