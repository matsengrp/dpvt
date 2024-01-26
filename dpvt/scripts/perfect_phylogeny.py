from ete3 import Tree
from itertools import (
    permutations as perms,
    combinations as combs,
    combinations_with_replacement as combs_r,
    product as prod,
)


class PerfectPhylogeny:
    """
    A class to determine multiple sequence alignments and mutation histories that form
    a perfect phylogeny for a given topology. This class supports varying levels of
    strictness for a perfect phylogeny. By default, we take a perfect phylogeny to be a
    topology with sequences on all nodes such that for each site and for each allowed
    character, the subgraph of nodes with the given character at the given site is
    connected. Options allow for requiring tip sequences to be unique, that no two
    columns in the sequence alignment (of all nodes) are identical, that every edge has
    at least one substition, and/or every non-terminal edge has at least one substition.

    The standard use case is to create an instance of the this class from a given ete3
    tree and call make_trees to generate perfect phylogenies on the tree.

    Attributes:
        tree (ete3.Tree): The topology.
        states (list): The allowed characters in sequences.
        state_count (int): The number of states.
        nodes (list of ete3.Trees): List of all nodes of the topology in preorder 
            traversal. The node indices used throughout this class are indices into this 
            list. Note the root always has node index 0.
        node_count (int): The nunber of nodes in the topology.
        node_index (dict): A dictionary mapping a node of the topology to its index in
            the list of nodes.
        leaf_indices (list): A list of leaf node indices in the list of nodes.
        leaf_count (int): The number of leaf nodes in the topology.
        state_permutations (dict): A dictionary mapping an integer r to the list of
            permutations using r elements of self.states.
        mutation_node_index_lists (list of tuples): Each inner tuple gives the indices 
            of nodes where mutations occur, such that the mutations can be chosen to 
            produce a perfect phylogeny.
        state_lists (list of lists): Each inner list is of length
            self.node_count, with the entry at a given node_index being an integer from
            0 to (self.state_count - 1); applying any bijection from
            {0, 1, ..., (self.state_count - 1)} to self.states yields a perfect
            phylogeny. The list at a given index is derived from the list of
            self.mutation_node_index_lists at the same index.
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
        Returns the substitutions for the parent of the specified to the node. This
        follows the same format as the node_sequence method.
        """
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
                    subs.append(f"{parent}{i}{child}")
        return "{" + "_".join(subs) + "}"

    def make_tree(
        self,
        label_seq,
        label_sub,
        state_lists_indices,
        perm_indices,
        unique_leaves=False,
    ):
        """
        Create a single new tree based on indices into self.state_lists and
        self.state_permutations. The nodes of the tree are optionally labelled with the
        sequence at that node and/or the substitions from the parent node. Returned is a
        pair consisting of the new tree and truth value for the tree passing additional
        requirements. Currently, the only supported additional requirement is that the
        leaf sequences are unique.
        """
        seq = lambda n: self.node_sequence(n, state_lists_indices, perm_indices)
        sub = lambda n: self.node_subs(n, state_lists_indices, perm_indices)

        tree = self.tree.copy()
        for node in tree.traverse(strategy="preorder"):
            node_index = node.node_index
            if label_seq:
                node.add_feature("sequence", seq(node_index))
            if label_sub:
                node.add_feature("subs", sub(node_index))

        is_valid = True
        if unique_leaves:
            if label_seq:
                leaf_seqs = {leaf.sequence for leaf in tree.get_leaves()}
            else:
                leaf_seqs = {seq(leaf.node_index) for leaf in tree.get_leaves()}
            is_valid &= len(leaf_seqs) == self.leaf_count

        return (tree, is_valid)

    def perms_for_states(self, state_list_indices):
        """
        Returns a list of lists. The i-th inner list consists of the valid indices into
        self.state_permutations[r], where r is the number of different states contained
        in the state_list specified by the i-th entry of state_list_indices.
        """
        r = lambda i: len(self.mutation_node_index_lists[i]) + 1
        return [
            list(range(len(self.state_permutations[r(i)]))) for i in state_list_indices
        ]

    def do_lists_mut_all_nodes(self, index_lists_indices):
        """
        Returns the truth value for the speficied state_lists (in self.state_lists)
        giving a perfect phylogeny where every non-root node has at least one mutation.
        """
        subbed_node_count = len(
            {
                node_index
                for i in index_lists_indices
                for node_index in self.mutation_node_index_lists[i]
            }
        )
        return subbed_node_count == self.node_count - 1

    def do_lists_mut_internal_nodes(self, index_lists_indices):
        """
        Returns the truth value for the speficied state_lists (in self.state_lists)
        giving a perfect phylogeny where every non-root non-leaf node has at least one
        mutation.
        """
        subbed_node_count = len(
            {
                node_index
                for i in index_lists_indices
                for node_index in self.mutation_node_index_lists[i]
                if node_index not in self.leaf_indices
            }
        )
        return subbed_node_count == self.node_count - self.leaf_count - 1

    def make_trees(
        self,
        use_seq=True,
        use_sub=True,
        unique_leaves=False,
        distinct_sites=False,
        sub_on_all_edges=False,
        sub_on_all_internal=False,
        max_sites=1,
    ):
        """
        Returns a generator for the perfect phylogenies, with nodes optionally labelled
        with sequences or substitions, meeting the given requirement. The generator
        produces all perfect phylogenies meeting the criteria, but without duplicates
        from permuting the order of sites in the sequences.
        """
        will_be_distinct = PerfectPhylogeny.paired_repeat
        next_tree = self.make_tree
        n = len(self.state_lists)
        trees = (
            t[0]
            for m in range(1, max_sites + 1)
            for state_lists in combs_r(range(n), m)
            if not sub_on_all_edges or self.do_lists_mut_all_nodes(state_lists)
            if not sub_on_all_internal or self.do_lists_mut_internal_nodes(state_lists)
            for perms in prod(*self.perms_for_states(state_lists))
            if not (distinct_sites and will_be_distinct(state_lists, perms))
            if (t := next_tree(use_seq, use_sub, state_lists, perms, unique_leaves))[1]
        )
        return trees

    @staticmethod
    def paired_repeat(list1, list2):
        """
        Returns the truth value of the two lists having a consecutive repeated value at
        the same index.
        """
        n = len(list1) - 1
        return any(
            (list1[i] == list1[i + 1] and list2[i] == list2[i + 1] for i in range(n))
        )

    @staticmethod
    def print_columns(tree):
        """Prints the columns of the sequence alignment for all nodes in the tree."""
        if not hasattr(tree, "sequence"):
            raise ValueError("The input tree does not have the sequence attribute.")
        cols = list(zip(*(n.sequence for n in tree.traverse(strategy="preorder"))))
        for i, c in enumerate(cols):
            print(f"column {i}: {c}")
        return None
