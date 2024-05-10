import numpy as np
from ete3 import Tree


class RandomPerfectPhylogeny:
    """
    This class takes an ete3 Tree and yields random perfect phylogenies for the
    underlying topology. The perfect phylogenies meet the same criteria as those given
    by the PerfectPhylogeny class. The main difference is this class returns a single
    random perfect phylogeny at a time, so various calculations are done on the fly
    rather than at initialization and there is no book keeping to enumerate all perfect
    phylogenies.
    """

    """Attributes:
        all_node_count (int): The length of all_nodes.
        all_nodes (list): A pre-order list of the nodes of the input tree, with all
            leaves appearing after the internal nodes. The index of a node in this list
            is the so-called node_index of the node.
        bad_root_patterns (list): A list of the mutation node index sets that are near
            the root and do not meet the requirement of a perfect phylogeny. 
        bases (tuple): The nucleotide bases 'A', 'G', 'C', and 'T'.
        first_leaf_index (int): The first index position of a leaf node in all_nodes.
        mut_counts (list): A list with the number of mutations at each node. The entry 
            at index i corresponds not to the node with node_index i, but to the node
            whose node_index is at index i of node_indices.
        mut_selections (list): A list of mutation node index sets that form the current
            random perfect phylogeny.
        no_perms (boolean): When True, there is no randomness in assigning nucleotide 
            bases for a mutation node index set. Specifically, each site corresponds to
            a set of substitutions on at most 3 non-root nodes. Suppose these three 
            nodes are x, y, and z, with the order following the specific pre-order 
            traversal stored in the all_nodes attribute. With no randomization, the 
            root, x, y, and z are assigned 'A', 'G', 'C', 'T', in that order. This can 
            be useful when testing.
        node_count (int): The length of node_indices.
        node_index_to_count_index (dict): The reverse map taking a node_index to the 
            appropriate index of mut_counts and node_indices.
        node_indices (tuple): The node indices of the nodes on which we require a 
            substitution (either all internal nodes or all non-root nodes). While node
            indices are assigned to nodes in a specific order, this tuple of node 
        node_sequences (dict): A dictionary mapping a node index to the current sequence
            for the node.
        node_substitutions (dict): A dictionary mapping a node index to the current 
            substitutions for the node.
        rng (numpy.random._generator.Generator): The random number generator used for 
            all random choices.
        size_probs (list): A length 3 list with the probabilities of selecting a set of
            mutations on 1, 2, or 3 nodes at one site.
        tree (ete3.Tree): The input tree.
    """

    def __init__(
        self,
        tree,
        edges="internal",
        no_permutations=False,
        size_probs=None,
        random_seed=None,
    ):
        self.tree = tree.copy()
        self.no_perms = no_permutations
        self.rng = np.random.default_rng(random_seed)
        self.bases = ("A", "G", "C", "T")
        self.set_node_data(edges)
        self.set_bad_root_patterns()
        self.set_size_probs(size_probs)

    def make_random_perfect_phylogeny(self, use_seq=True, use_sub=False):
        """
        Returns a new ete3 Tree instance with the topology of the original tree and with
        all nodes labelled (by sequence or substitutions) to make a perfect phylogeny.
        """
        self.set_internal_state()
        self.set_mutation_selection()
        self.set_sequence_data()
        tree = self.make_labelled_tree(use_seq=use_seq, use_sub=use_sub)
        return tree

    def set_node_data(self, edges):
        """
        Sets various attributes about tree nodes. Called only at initialization.
        """
        self.all_nodes = []
        next_index = 0
        for node in self.tree.traverse(strategy="preorder"):
            if not node.is_leaf():
                node.add_feature("node_index", next_index)
                self.all_nodes.append(node)
                next_index += 1
        self.first_leaf_index = next_index
        for node in self.tree.get_leaves():
            node.add_feature("node_index", next_index)
            self.all_nodes.append(node)
            next_index += 1
        self.all_node_count = next_index

        if edges == "internal":
            self.node_indices = range(1, self.first_leaf_index)
            self.node_count = self.first_leaf_index - 1
        elif edges == "all":
            self.node_indices = range(1, self.all_node_count)
            self.node_count = self.all_node_count - 1
        else:
            raise ValueError("Edges must be 'all' or 'internal'.")

        return None

    def set_bad_root_patterns(self):
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

    def set_size_probs(self, size_probs=None):
        """
        Set the size_probs attribute, which is a vector of probabilities. The
        probabilities are for selection a mutation node index set (the nodes with
        substitutions at a common site) with 1, 2, or 3 nodes in total. By default, the
        probabilities are calculated so that all mutation node index sets are equally
        likely, which means a mutation node index set with a single node is highly
        unlikely when there are many nodes in the tree. Another reasonable option is
        [1/3,1/3,1/3].
        """
        if size_probs is None:
            n = self.all_node_count - 2
            c1, c2, c3 = 1, n, (n * (n - 1)) // 2
            total = c1 + c2 + c3
            self.size_probs = (c1 / total, c2 / total, c3 / total)
        else:
            self.size_probs = size_probs

    def set_internal_state(self):
        """
        Initializes and clear various internal state attributes used in determining a
        perfect phylogeny.
        """
        self.node_indices = tuple(self.rng.permutation(self.node_indices))
        self.node_index_to_count_index = {n: i for i, n in enumerate(self.node_indices)}
        self.mut_selections = []
        self.mut_counts = [0] * self.node_count
        return None

    def set_mutation_selection(self):
        """
        Sets the self.mut_selections to a random list of valid mutation node index sets
        that form a perfect phylogeny. Assumes various internal state attributes are
        clear.
        """
        there_are_nodes_without_subs = True
        while there_are_nodes_without_subs:
            # Get the next node index requiring a mutation
            n_index = self.node_indices[self.mut_counts.index(0)]
            keep_trying = True
            while keep_trying:
                mut_set = self.make_random_mutation_set(n_index)
                keep_trying = self.any_previous_mut_set_superfluous(mut_set)
            self.mut_selections.append(mut_set)
            self.increment_mutation_counts(mut_set)
            there_are_nodes_without_subs = 0 in self.mut_counts
        return None

    def alter_mutation_counts(self, mut_set, i):
        for n in mut_set.intersection(self.node_indices):
            self.mut_counts[self.node_index_to_count_index[n]] += i

    def increment_mutation_counts(self, mut_set):
        self.alter_mutation_counts(mut_set, 1)

    def decrement_mutation_counts(self, mut_set):
        self.alter_mutation_counts(mut_set, -1)

    def set_sequence_data(self):
        """
        Sets the node_sequences and node_substitutions attributes to contain the
        sequences and substitutions for all nodes in the tree, based on the
        mut_selections attribute.
        """
        if self.no_perms:
            perms = [self.bases for _ in self.mut_selections]
        else:
            perms = [self.rng.permutation(self.bases) for _ in self.mut_selections]
        node_sequences = {
            n_idx: [perms[s_idx][0] for s_idx in range(len(self.mut_selections))]
            for n_idx in range(self.all_node_count)
        }
        node_subs = {n_idx: [] for n_idx in range(self.all_node_count)}

        for s_idx, mut_node_indices in enumerate(self.mut_selections):
            for i, n_idx in enumerate(mut_node_indices, 1):
                node_sequences[n_idx][s_idx] = perms[s_idx][i]
            for node in self.all_nodes:
                if node.is_root():
                    continue
                n_idx = node.node_index
                p_idx = node.up.node_index
                if n_idx in mut_node_indices:
                    p_base = node_sequences[p_idx][s_idx]
                    n_base = node_sequences[n_idx][s_idx]
                    node_subs[n_idx].append(f"{p_base}{s_idx}{n_base}")
                else:
                    node_sequences[n_idx][s_idx] = node_sequences[p_idx][s_idx]

        self.node_sequences = node_sequences
        self.node_substitutions = node_subs
        return None

    def no_bad_patterns(self, mut_node_indices):
        """
        Returns the truth value of the set of mutated node indices avoiding the
        patterns:
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
        , where x, y, z are the nodes (in any order).
        """
        if len(mut_node_indices) == 1:
            return True
        else:
            if len(mut_node_indices) == 3:
                mut_node_indices = sorted(mut_node_indices)
                n1, n2, n3 = mut_node_indices
                y, z = self.all_nodes[n2], self.all_nodes[n3]
                is_first_pattern = y.up.node_index == z.up.node_index == n1
            else:
                is_first_pattern = False
            is_root_pattern = any(
                bad.issubset(mut_node_indices) for bad in self.bad_root_patterns
            )
            return not (is_first_pattern or is_root_pattern)

    def make_random_mutation_set(self, node_index):
        """
        Returns a valid mutation node index set containing the given node index. The
        probability of this set containing 1, 2, or 3 node indices is given by the
        size_probs attribute.
        """
        how_many_more = self.rng.choice(3, p=self.size_probs)
        # how_many_more = self.rng.choice(3)
        if how_many_more == 0:
            return {node_index}

        other_node_indices = tuple(
            (*range(1, node_index), *range(node_index + 1, self.all_node_count))
        )
        mut_set_is_valid = False
        while not mut_set_is_valid:
            more = self.rng.choice(other_node_indices, size=how_many_more)
            mut_set = {node_index, *more}
            mut_set_is_valid = self.no_bad_patterns(mut_set)

        return mut_set

    def make_labelled_tree(self, use_seq=True, use_sub=False):
        """
        Returns a new ete3 Tree instance with nodes labelled according to the
        node_sequences and node_substitutions attributes.
        """
        tree = self.tree.copy()
        for node in tree.traverse():
            if use_seq:
                the_sequence = "".join(self.node_sequences[node.node_index])
                node.add_feature("sequence", the_sequence)
            if use_sub:
                the_sub = "{" + "_".join(self.node_substitutions[node.node_index]) + "}"
                node.add_feature("subs", the_sub)
        return tree

    def any_previous_mut_set_superfluous(self, new_mut_set):
        """
        Returns the truth value for the mutation node index set to cause a previously
        selected mutation set to become unnecessary for all required nodes to have
        mutations.
        """
        self.increment_mutation_counts(new_mut_set)
        old_not_needed = False
        for old_mut_set in self.mut_selections:
            self.decrement_mutation_counts(old_mut_set)
            old_not_needed = all(
                self.mut_counts[self.node_index_to_count_index[n]] > 0
                for n in old_mut_set.intersection(self.node_indices)
            )
            self.increment_mutation_counts(old_mut_set)
            if old_not_needed:
                break
        self.decrement_mutation_counts(new_mut_set)
        return old_not_needed
