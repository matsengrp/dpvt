from ete3 import Tree
from itertools import (
    permutations as perms,
    combinations as combs,
    product as prod,
    chain,
)
from dpvt.scripts.minimal_covers import MinimalCovers
from math import ceil


class PerfectPhylogeny:
    """
    A class to determine multiple sequence alignments and mutation histories that form a
    perfect phylogeny for a given tree topology. This class supports varying levels of
    strictness for a perfect phylogeny. By default, we take a perfect phylogeny to be a
    topology with sequences on all nodes such that 1) for each site and for each
    character appearing at the site, the subgraph of nodes with the given character at
    the given site is connected and contains a leaf node; 2) every non-terminal edge
    has at least one substitution; and 3) omitting any single site violates 2). Options
    allow for requiring in 2) that every edge has a substitution and for requiring tip
    sequences to be unique (currently not supported).

    The standard use case is to create an instance of the this class from a given ete3
    Tree and call make_phylogenies to get a generator for perfect phylogenies on the
    tree, or call make_random_phylogeny to get a single perfect phylogeny.

    Attributes:
        bad_root_patterns (set): The set of tuples of node indices not allowed as
            mutations near the root.
        cherry_index_pairs (set): The set of pairs of leaf indices (leaf1, leaf2),
            where the two leaves are siblings and leaf1.node_index < leaf2.node_index.
        internal_node_count (int): The number of non-root non-leaf nodes.
        internal_node_indices (set): The set of internal node indices.
        leaf_count (int): The number of leaf nodes in the topology.
        leaf_indices (set): The set of leaf node indices.
        mutation_internal_node_index_sets (tuple of sets): The entries of
            mutation_node_index_sets restricted to internal nodes.
        mutation_leaf_node_index_sets (tuple of sets): The entries of
            mutation_node_index_sets restricted to leaf nodes.
        mutation_node_index_sets (tuple of sets): Each inner set gives the indices of
            nodes where mutations occur, such that the number of mutations is at most
            (state_count - 1). This means the mutations can be chosen to produce a
            perfect phylogeny.
        node_count (int): The number of nodes in the topology.
        node_indices (set): The set of node indices.
        nodes (tuple of ete3.Trees): Tuple of all nodes of the topology in preorder
            traversal, with all internal nodes appearing before any leaf nodes. The
            node indices used throughout this class are indices into this tuple. Note
            the root node always has index 0.
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

    def __init__(self, tree: Tree, states: tuple[str] = ("A", "G", "C", "T")):
        if len(tree.get_leaves()) <= 2:
            raise ValueError("Please supply a tree with more than two leaves.")
        if len(states) > 4:
            raise NotImplementedError(
                "We do not currently support more than four letters."
            )

        self.tree = tree.copy()
        self.states = states
        self.state_count = len(self.states)
        self.state_permutations = {
            r: tuple(perms(self.states, r)) for r in range(1, self.state_count + 1)
        }

        nodes = []
        next_index = 0
        for node in self.tree.traverse(strategy="preorder"):
            if not node.is_leaf():
                node.add_feature("node_index", next_index)
                nodes.append(node)
                next_index += 1
        self.internal_node_indices = set(range(1, next_index))
        self.internal_node_count = next_index - 1
        first_leaf_index = next_index
        for node in self.tree.get_leaves():
            node.add_feature("node_index", next_index)
            nodes.append(node)
            next_index += 1
        self.leaf_indices = set(range(first_leaf_index, next_index))
        self.leaf_count = next_index - first_leaf_index
        self.nodes = tuple(nodes)
        self.node_count = next_index
        self.node_indices = set(range(self.node_count))

        self.make_bad_root_patterns()
        self.make_cherry_index_pairs()
        self.make_mutation_index_sets()
        self.make_mutation_internal_node_index_sets()
        self.make_mutation_leaf_node_index_sets()
        self.make_state_tuples()

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
        self.bad_root_patterns = [{left_index, right_index}]
        if not left.is_leaf():
            ll_index, lr_index = (x.node_index for x in left.children)
            self.bad_root_patterns.append({right_index, ll_index, lr_index})
        if not right.is_leaf():
            rl_index, rr_index = (x.node_index for x in right.children)
            self.bad_root_patterns.append({left_index, rl_index, rr_index})

        return None

    def no_bad_patterns(self, mut_node_indices):
        """
        Returns the truth value of the tuple of mutated node indices (assumed to be in
        ascending order) avoiding the patterns:
               /-y
            -x|
               \-z
        ,
                  /-x
            -root|
                  \-y
        , and
                              /-y
                  /-some node|
            -root|            \-z
                  \-x
        , where x, y, z are the node indices (in any order).
        """
        if len(mut_node_indices) <= 1:
            return True
        else:
            n1 = mut_node_indices[0]
            n1_child_indices = (x.node_index for x in self.nodes[n1].children)
            n2_n3 = set(mut_node_indices[1:])
            is_first_pattern = n1 in self.internal_node_indices and n2_n3.issuperset(
                n1_child_indices
            )
            # Because mut_node_indices is in ascending order and self.nodes is in
            # preorder, we only need to check for n1 being the parent of n2 and n3.
            # If there are only two node indices instead of three, this pattern fails.

            is_root_pattern = any(
                bad.issubset(mut_node_indices) for bad in self.bad_root_patterns
            )

            return not (is_first_pattern or is_root_pattern)

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
        """Initialize self.cherry_index_pairs. This is the set of pairs of leaf indices
        (leaf1, leaf2), where the two leaves are siblings and
        leaf1.node_index < leaf2.node_index."""
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
        for node in self.nodes:
            index = node.node_index
            if not (index in mutation_node_indices or node.is_root()):
                parent_index = node.up.node_index
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

    def make_phylogeny(self, label_seq, label_sub, state_tuples_indices, perm_indices):
        """
        Create a single new phylogeny based on indices into self.state_tuples and
        self.state_permutations. The nodes of the tree are optionally labelled with the
        sequence at that node and/or the substitutions from the parent node.
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

    def perms_for_states(self, state_tuples_indices, skip_perms=False):
        """
        Returns a tuple of tuples. The i-th inner tuple consists of the valid indices
        into self.state_permutations[r], where r is the number of different states
        contained in the entry of self.state_tuples at index state_tuples_indices[i].
        """
        r = lambda i: len(self.mutation_node_index_sets[i]) + 1
        if skip_perms:
            return tuple((0,) for i in state_tuples_indices)
        else:
            return tuple(
                tuple(range(len(self.state_permutations[r(i)])))
                for i in state_tuples_indices
            )

    def are_cherries_distinct(self, index_sets_indices):
        indices = (self.mutation_leaf_node_index_sets[i] for i in index_sets_indices)
        subbed_leaves = set(chain(*indices))
        return all(
            (
                left in subbed_leaves or right in subbed_leaves
                for left, right in self.cherry_index_pairs
            )
        )

    ###better name

    def make_random_phylogeny(
        self,
        use_seq=True,
        use_sub=False,
        unique_leaves=False,
        sub_on_all_edges=False,
    ):
        """
        Returns a random perfect phylogeny meeting the given criteria.
        Currently we don't randomize the site permutations.
        """
        return next(
            self.make_phylogenies(
                use_seq=use_seq,
                use_sub=use_sub,
                unique_leaves=unique_leaves,
                sub_on_all_edges=sub_on_all_edges,
                shuffle=True,
                skip_perms=True,
            )
        )

    def make_phylogenies(
        self,
        use_seq=True,
        use_sub=False,
        unique_leaves=False,
        sub_on_all_edges=False,
        min_sites=1,
        max_sites=None,
        skip_perms=False,
        shuffle=False,
    ):
        """
        Returns a generator for the perfect phylogenies meeting the given criteria.
        """
        # The minimum number of sites to produce a perfect phylogeny is
        # ceil(number of edges required subs / maximum number of mutations in a site),
        # which is either ceil((self.leaf_count - 2) / (self.state_count - 1))
        # or ceil((self.node_count - 1) / (self.state_count - 1)).
        # Currently min and max sites are handled by filtering after making the
        # generator, which can be slow. In particular, a bad choice of max sites would
        # mean enumerating all perfect phylogenies just to find none have the allowed
        # number of sites.
        if unique_leaves:
            raise NotImplementedError("Code isn't ready yet.")

        if sub_on_all_edges:
            site_min = ceil((self.node_count - 1) / (self.state_count - 1))
        else:
            site_min = ceil((self.leaf_count - 2) / (self.state_count - 1))
        site_min = max(site_min, min_sites)
        if max_sites is not None:
            if max_sites < site_min:
                raise ValueError(f"Max_sites must be at least {site_min}.")
            length_check = lambda s: site_min <= len(s) <= max_sites
        else:
            length_check = lambda s: site_min <= len(s)
        which_edges = "all" if sub_on_all_edges else "internal"

        state_tuple_indices_gen = (
            state_tuple_indices
            for state_tuple_indices in MinimalCovers(
                self, which_edges, shuffle_indices=shuffle
            )
            if length_check(state_tuple_indices)
        )

        trees = (
            self.make_phylogeny(use_seq, use_sub, state_tuples_indices, perm_indices)
            for state_tuples_indices in state_tuple_indices_gen
            for perm_indices in prod(
                *self.perms_for_states(state_tuples_indices, skip_perms)
            )
        )
        return trees

    @staticmethod
    def print_alignment(tree):
        """Prints the columns of the sequence alignment for all nodes in the tree."""
        if not hasattr(tree, "sequence"):
            raise ValueError("The input tree does not have the sequence attribute.")
        cols = list(zip(*(n.sequence for n in tree.traverse(strategy="preorder"))))
        for i, c in enumerate(cols):
            print(f"site {i}: {c}")
        return None
