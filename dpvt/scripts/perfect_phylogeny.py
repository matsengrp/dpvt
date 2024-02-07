from ete3 import Tree
from itertools import (
    permutations as perms,
    combinations as combs,
    combinations_with_replacement as combs_r,
    product as prod,
    repeat,
    chain,
)
from time import time
import numpy as np


class PerfectPhylogeny:
    """
    A class to determine multiple sequence alignments and mutation histories that form
    a perfect phylogeny for a given topology. This class supports varying levels of
    strictness for a perfect phylogeny. By default, we take a perfect phylogeny to be a
    topology with sequences on all nodes such that for each site and for each allowed
    character, the subgraph of nodes with the given character at the given site is
    connected and contains a leaf node. Options allow for requiring tip sequences to be
    unique, that no two columns in the sequence alignment (of all nodes) are identical,
    that every edge has at least one substition, and/or every non-terminal edge has at
    least one substition.

    The standard use case is to create an instance of the this class from a given ete3
    tree and call make_trees to generate perfect phylogenies on the tree.

    Attributes:
        bad_root_patterns (set tuples): The tuples of node indices not allowed as
            mutations near the root.
        cherry_index_pairs (set of pairs): The set of pairs of leaf indices
            (leaf1, leaf2), where the two leaves are siblings and
            leaf1.node_index < leaf2.node_index.
        internal_node_count (int): The number of non-root non-leaf nodes.
        internal_node_indices (set): The set of internal node indices, which is a
            subset of the values of the dictionary node_index.
        leaf_count (int): The number of leaf nodes in the topology.
        leaf_indices (set): The set of leaf node indices, which is a subset of the
            values of the dictionary node_index.
        mutation_node_index_sets (tuple of sets): Each inner set gives the indices
            of nodes where mutations occur, such that the number of mutations is at most
            (state_count - 1). This means the mutations can be chosen to produce a
            perfect phylogeny.
        mutation_internal_node_index_sets (tuple of sets): The entries of
            mutation_node_index_sets restricted to internal nodes.
        mutation_leaf_node_index_sets (tuple of sets): The entries of
            mutation_node_index_sets restricted to leaf nodes.
        node_count (int): The number of nodes in the topology.
        node_index (dict): A dictionary mapping a node of the topology to its index in
            the list of nodes. Note the root node always has index 0.
        nodes (tuple of ete3.Trees): Tuple of all nodes of the topology in preorder
            traversal. The node indices used throughout this class are indices into this
            tuple. Note the root node always has index 0.
        state_count (int): The number of states.
        state_permutations (dict): A dictionary mapping an integer r to the tuple of
            permutations using r elements of self.states. The states are used to convert
            state-placeholders to valid states.
        state_tuples (tuple of tuple): Each inner tuple is of length self.node_count,
            with the entry at a given node_index being an integer from 0 to
            (state_count - 1); applying any bijection from
            {0, 1, ..., (self.state_count - 1)} to self.states yields a labelled
            topology satisfying the subgraph criteria of a perfect phylogeny. The inner
            tuple at a given index is derived from the set at the same index in
            self.mutation_node_index_sets.
        states (tuple): The allowed characters in sequences.
        tree (ete3.Tree): The topology.
    """

    def __init__(self, tree, states=("A", "G", "C", "T")):
        if len(tree.get_leaves()) <= 2:
            raise ValueError("Please supply a tree with more than two leaves.")
        if len(states) > 4:
            raise NotImplementedError(
                "We do not currently support more than four letters."
            )

        self.tree = tree.copy()
        self.states = states
        self.state_count = len(self.states)
        self.nodes = tuple(self.tree.traverse(strategy="preorder"))
        self.node_count = len(self.nodes)
        self.node_index = dict(zip(self.nodes, range(self.node_count)))
        self.leaf_indices = {self.node_index[leaf] for leaf in self.tree.get_leaves()}
        self.leaf_count = len(self.leaf_indices)
        self.internal_node_indices = {
            i for i in self.node_index.values() if i not in self.leaf_indices and i > 0
        }
        self.internal_node_count = self.node_count - self.leaf_count - 1
        self.state_permutations = {
            r: tuple(perms(self.states, r)) for r in range(1, self.state_count + 1)
        }

        for node in self.nodes:
            node.add_feature("node_index", self.node_index[node])
        self.make_bad_root_patterns()
        self.make_cherry_index_pairs()
        self.make_mutation_index_sets()
        self.make_mutation_internal_node_index_sets()
        self.make_mutation_leaf_node_index_sets()
        self.make_state_tuples()

        # CJS: This is work in progress for random sampling. Ignore for now.
        # self.rng = np.random.default_rng()
        # self.make_indices()
        # self.shuffle_indices()

    # def make_indices(self):
    #    """...
    #    first coordinate is needed if we want to have order of sites consistent with other thing
    #    """
    #    N = self.node_count - 1
    #    combo_counts = [1, N, N * (N - 1) // 2, N * (N - 1) * (N - 2) // 6]
    #    offsets = [0, *np.cumsum(combo_counts)[:-1]]
    #    perm_counts = [4, 12, 24, 24]
    #    self.indices = [
    #        (i + 1, offsets[i] + j, k)
    #        for i in range(4)
    #        for j in range(combo_counts[i])
    #        for k in range(perm_counts[i])
    #    ]
    #    return None
    #
    # def shuffle_indices(self):
    #    self.rng.shuffle(self.indices)
    #    return None

    def make_bad_root_patterns(self):
        """
        Initialize self.bad_root_patterns to contain the disallowed mutation patterns:
                              /-y
                  /-some node|
            -root|            \-z
                  \-x
        and
                  /-x
            -root|
                  \-y
        where x, y, and z have mutations.
        """
        left, right = self.tree.children
        left_index = left.node_index
        right_index = right.node_index
        self.bad_root_patterns = {tuple(sorted((left_index, right_index)))}
        if not left.is_leaf():
            ll_index, lr_index = map(lambda x: x.node_index, left.children)
            self.bad_root_patterns.add(tuple(sorted((right_index, ll_index, lr_index))))
        if not right.is_leaf():
            rl_index, rr_index = map(lambda x: x.node_index, right.children)
            self.bad_root_patterns.add(tuple(sorted((left_index, rl_index, rr_index))))
        return None

    def no_bad_patterns(self, node_indices):
        """
        Returns the truth value of the tuple of node indices (assumed to be in ascending
        order) avoiding the patterns:
               /-y
            -x|
               \-z
        and
                              /-y
                  /-some node|
            -root|            \-z
                  \-x
        ,where x, y, z are the node indices (in any order).
        """
        if len(node_indices) <= 2:
            return True
        else:
            n1 = node_indices[0]
            n2_n3 = set(node_indices[1:])
            pattern1 = n1 in self.internal_node_indices and n2_n3.issuperset(
                self.nodes[n1].children
            )
            # Because node_indices is in ascending order and self.nodes is in preorder,
            # we only need to check for n1 being the parent of n2 and n3.

            pattern2 = node_indices in self.bad_root_patterns

            return not (pattern1 or pattern2)

    def make_mutation_index_sets(self):
        """Initialize self.mutation_node_index_sets."""
        self.mutation_node_index_sets = tuple(
            set(mutated_node_indices)
            for num_subs in range(0, self.state_count)
            for mutated_node_indices in combs(range(1, self.node_count), num_subs)
            if self.no_bad_patterns(mutated_node_indices)
        )
        return None

    def make_mutation_internal_node_index_sets(self):
        """Initialize self.mutation_internal_node_index_sets."""
        self.mutation_internal_node_index_sets = tuple(
            map(self.internal_node_indices.intersection, self.mutation_node_index_sets)
        )
        return None

    def make_mutation_leaf_node_index_sets(self):
        """Initialize self.mutation_leaf_node_index_sets."""
        self.mutation_leaf_node_index_sets = tuple(
            map(self.leaf_indices.intersection, self.mutation_node_index_sets)
        )
        return None

    def make_state_tuples(self):
        """Initialize self.state_tuples."""
        self.state_tuples = tuple(
            map(self.mutation_to_state_tuple, self.mutation_node_index_sets)
        )
        return None

    def make_cherry_index_pairs(self):
        """Initialize self.cherry_index_pairs."""
        s_index = lambda node_index: self.nodes[node_index].get_sisters()[0].node_index
        self.cherry_index_pairs = {
            (leaf_index, sister_index)
            for leaf_index in self.leaf_indices
            if (
                (sister_index := s_index(leaf_index)) > leaf_index
                and sister_index in self.leaf_indices
            )
        }
        return None

    def mutation_to_state_tuple(self, mutation_node_indices):
        """
        Make an entry for self.state_tuples from one of self.mutation_node_index_sets.
        """
        state_list = [0] * self.node_count
        for i, j in enumerate(mutation_node_indices, 1):
            state_list[j] = i
        for node, index in self.node_index.items():
            if not (index in mutation_node_indices or node.is_root()):
                parent_index = self.node_index[node.up]
                state_list[index] = state_list[parent_index]
        return tuple(state_list)

    def node_sequence(self, node_index, state_tuples_indices, perm_indices):
        """
        Returns the sequence for the node at the given index, based on the specified
        state tuples and permutations. Specifically, the character at site i of the
        sequence is obtained by applying the j-th permutation (of appropriate size) to
        the entry of the k-th state tuple corresponding to the node, where j is the i-th
        entry of perm_indices and k is the i-th entry of state_tuple_indices.
        """
        r = lambda i: len(self.mutation_node_index_sets[i]) + 1
        perm_fn = lambda r, i, x: self.state_permutations[r][i][x]
        return "".join(
            (
                perm_fn(r(k), j, self.state_tuples[k][node_index])
                for j, k in zip(perm_indices, state_tuples_indices)
            )
        )

    def node_subs(self, node_index, state_tuples_indices, perm_indices):
        """
        Returns the substitutions for the parent of the specified to the node. This
        follows the same format as the node_sequence method.
        """
        r = lambda i: len(self.mutation_node_index_sets[i]) + 1
        perm_fn = lambda r, i, x: self.state_permutations[r][i][x]
        subs = []
        if node_index != 0:
            parent_index = self.nodes[node_index].up.node_index
            zipped = zip(state_tuples_indices, perm_indices)
            for i, (st_index, perm_index) in enumerate(zipped):
                parent_site = self.state_tuples[st_index][parent_index]
                node_site = self.state_tuples[st_index][node_index]
                if parent_site != node_site:
                    parent = perm_fn(r(st_index), perm_index, parent_site)
                    child = perm_fn(r(st_index), perm_index, node_site)
                    subs.append(f"{parent}{i}{child}")
        return "{" + "_".join(subs) + "}"

    def make_tree(self, label_seq, label_sub, state_tuples_indices, perm_indices):
        """
        Create a single new tree based on indices into self.state_tuples and
        self.state_permutations. The nodes of the tree are optionally labelled with the
        sequence at that node and/or the substitions from the parent node.
        """
        seq = lambda n: self.node_sequence(n, state_tuples_indices, perm_indices)
        sub = lambda n: self.node_subs(n, state_tuples_indices, perm_indices)

        tree = self.tree.copy()
        for node in tree.traverse(strategy="preorder"):
            node_index = node.node_index
            if label_seq:
                node.add_feature("sequence", seq(node_index))
            if label_sub:
                node.add_feature("subs", sub(node_index))

        return tree

    def perms_for_states(self, state_tuples_indices):
        """
        Returns a tuple of tuples. The i-th inner tuple consists of the valid indices into
        self.state_permutations[r], where r is the number of different states contained
        in the entry of self.state_tuples at index state_tuples_indices[i].
        """
        r = lambda i: len(self.mutation_node_index_sets[i]) + 1
        return tuple(
            tuple(range(len(self.state_permutations[r(i)])))
            for i in state_tuples_indices
        )

    def are_there_subs_on_all_nodes(self, index_sets_indices):
        """
        Returns the truth value for the specified sets of mutations in
        self.mutation_node_index_sets (equivalently, the entries in self.state_tuples)
        giving a perfect phylogeny where every non-root node has at least one mutation.
        """
        node_indices = (self.mutation_node_index_sets[i] for i in index_sets_indices)
        subbed_node_count = len(set(chain(*node_indices)))
        return subbed_node_count == self.node_count - 1

    def are_there_subs_on_all_internal_nodes(self, index_sets_indices):
        """
        Returns the truth value for the specified sets of mutations in
        self.mutation_node_index_sets (equivalently, the entries in self.state_tuples)
        giving a perfect phylogeny where every non-root non-leaf node has at least one
        mutation.
        """
        node_indices = (
            self.mutation_internal_node_index_sets[i] for i in index_sets_indices
        )
        subbed_node_count = len(set(chain(*node_indices)))
        return subbed_node_count == self.internal_node_count

    def are_cherries_distinct(self, index_sets_indices):
        indices = (self.mutation_leaf_node_index_sets[i] for i in index_sets_indices)
        subbed_leaves = set(chain(*indices))
        return all(
            (
                left in subbed_leaves or right in subbed_leaves
                for left, right in self.cherry_index_pairs
            )
        )

    def make_trees(
        self,
        use_seq=True,
        use_sub=True,
        unique_leaves=False,
        distinct_sites=False,
        sub_on_all_edges=False,
        sub_on_all_internal=False,
        min_sites=1,
        max_sites=1,
    ):
        """
        Returns a generator for the perfect phylogenies, with nodes optionally labelled
        with sequences or substitions, meeting the given requirement. The generator
        produces all perfect phylogenies meeting the criteria, but without duplicates
        from permuting the order of sites in the sequences.
        """
        sites_will_repeat = PerfectPhylogeny.paired_repeat
        next_tree = self.make_tree
        n = len(self.state_tuples)

        if sub_on_all_edges:
            state_checks = self.are_there_subs_on_all_nodes
        elif sub_on_all_internal:
            if unique_leaves:
                state_checks = lambda x: (
                    self.are_there_subs_on_all_internal_nodes(x)
                    and self.are_cherries_distinct(x)
                )
            else:
                state_checks = self.are_there_subs_on_all_internal_nodes
        else:
            raise NotImplementedError("Annoying case.")

        trees = (
            next_tree(use_seq, use_sub, state_tuples_indices, perm_indices)
            for m in range(min_sites, max_sites + 1)
            for state_tuples_indices in combs_r(range(n), m)
            if state_checks(state_tuples_indices)
            for perm_indices in prod(*self.perms_for_states(state_tuples_indices))
            if not (
                distinct_sites and sites_will_repeat(state_tuples_indices, perm_indices)
            )
        )
        return trees

    # def sample_indices(self, n, replace):
    #    # Sampling perfect phylos without replacement would require bookkeeping at this step.
    #    index_samples = sorted(
    #        self.rng.choice(
    #            self.indices, size=n, replace=replace, shuffle=False
    #        ).tolist()
    #    )
    #    _, state_lists_indices, permutation_indices = zip(*index_samples)
    #    return state_lists_indices, permutation_indices
    #
    # def random_tree_gen(
    #    self,
    #    use_seq=True,
    #    use_sub=True,
    #    unique_leaves=True,
    #    distinct_sites=True,
    #    sub_on_all_edges=False,
    #    sub_on_all_internal=True,
    #    sites=1,
    #    max_tries=None,
    # ):
    #    """
    #    ...
    #    """
    #    # Catch the empty generator case and return an empty tuple.
    #    all_trees = self.make_trees(
    #        use_seq,
    #        use_sub,
    #        unique_leaves,
    #        distinct_sites,
    #        sub_on_all_edges,
    #        sub_on_all_internal,
    #        sites,
    #        sites,
    #    )
    #    try:
    #        next(all_trees)
    #    except StopIteration:
    #        return ()
    #
    #    sample_indices = lambda: self.sample_indices(sites, replace=not distinct_sites)
    #    next_tree = lambda x, y: self.make_tree(use_seq, use_sub, x, y, unique_leaves)
    #    edge_checks = lambda x: all(
    #        (
    #            not sub_on_all_edges or self.do_lists_mut_all_nodes0(x),
    #            not sub_on_all_internal or self.do_lists_mut_internal_nodes(x),
    #        )
    #    )
    #    tries = repeat(0) if max_tries is None else repeat(0, max_tries)
    #    return (
    #        tree[0]
    #        for _ in tries
    #        if edge_checks((states_and_perms := sample_indices())[0])
    #        if (tree := next_tree(*states_and_perms))[1]
    #    )

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
