import numpy as np


class MinimalCovers:
    """
    A helper class for PerfectPhylogeny. This class takes in a PerfectPhylogeny instance
    and provides a generator for lists of indices into PerfectPhylogeny.state_tuples.
    Each list of indices,
        1) provides mutations at all internal edges (optionally, all edges), and
        2) is minimal in that omitting any single index violates 1).
    Note this is not the same as a minimal perfect phylogeny, where minimality is about
    contracting edges.

    When shuffle_indices=True at initalization, various lists of indices are permuted.
    The generator will produce the same minimal covers as usual, but in a different
    order. Only the first minimal cover should be considered random, but even that
    sample is not drawn uniformly from all minimal covers.
    """

    # Need to think about how to make the first sample more of a uniform draw...
    # Currently some perfect phylogenies are hot spots, but it is much more noticeable
    # when requiring mutations on all non-terminal edges than all edges.

    # For later. When just making a single random minimal cover, I should be able to
    # speed this up by taking only the topology as input and generating mutation node
    # index sets on the fly.

    """The indices are returned as a sorted list. The order in which the lists are
    generated is as follows.

    Suppose the nodes n_1, n_2, ..., n_N are indexed by 1, 2, ..., N. Let M_i denote the
    set of indices into PerfectPhylogeny.state_tuples that provide a mutation at the 
    node n_i. That is,
        j is in M_i iff i is in PerfectPhylogeny.mutation_node_index_sets[j].

    Each list of indices satifying 1) and 2) may be viewed an as element of the
    Cartesian product: PowerSet(M_1) x PowerSet(M_2) x ... x PowerSet(M_N).
    Specifically, they are the tuples (A_1, A_2, ..., A_N) where
        a) no A_i is empty,
        b) for all i, j, and k: if j is in A_i and k is in
           PerfectPhylogeny.mutation_node_index_sets[j], then j is in A_k (i.e., if an 
           element of PerfectPhylogeny.mutation_node_index_sets is selected, then it is 
           recorded for all nodes to which it applies),
        c) for all j in U({A_1, A_2, ..., A_k}), there exists i such that A_i = [j].

    The ordering on M_i induces a ordering on PowerSet(M_i), which induces an ordering
    on the Cartesian product, which induces an ordering on the subset satisfying a), b),
    and c). The generator follows this ordering.

    Attributes:
        mut_counts (list): A list of integers, where the element at index i is the
            current number of mutations applied to a node. This list is in the same order
            as self.node_indices.     
        mut_node_index_sets (tuple of sets): This is either the attribute
            mutation_internal_node_index_sets or mutation_node_index_sets of the input
            PerfectPhylogeny.
        mut_selections (list): The current selection of indices into
            PerfectPhylogeny.mutation_node_index_sets.
        node_index_to_count_index (dict): A dictionary mapping a node index to its
            associated index in self.mut_counts, which is the positition of the node 
            index in self.node_indices. By default, the node index n is at positiion n-1.    
            When shuffle_indices is True, the order is random. 
        node_index_to_mutation_set_indices (dict of tuples): A dictionary mapping a node
            index to the tuple of indices into PerfectPhylogeny.mutation_node_index_sets
            that target the node. Be default, the tuple of indices is in ascending order
            by index value. When shuffle_indices is True, each tuple of indices is 
            randomly permuted.
        node_indices (tuple): A tuple of the node indices, being either the internal 
            nodes or all non-root nodes. By default, the node index n is at entry (n-1), 
            with the offset due to the root node. When shuffle_indices is True, the 
            ordering is random.
        targeted_node_indices (list): A list of node indices. The entry at index
            i specifies the node that needed a mutation when we appended the ith entry to 
            self.mut_selection.
    """

    def __init__(
        self,
        perfect_phylogenies,
        edges="internal",
        shuffle_indices=False,
        random_seed=None,
        # distinct_leaves=False,
        # min_sites=1,
        # max_sites=None,
    ):
        # self.min_sites = min_sites
        # self.max_sites = max_sites
        # self.site_count = 0
        pp = perfect_phylogenies

        if edges == "internal":
            self.mut_node_index_sets = pp.mutation_internal_node_index_sets
            self.node_indices = tuple(pp.internal_node_indices)
            self.node_count = pp.internal_node_count
        elif edges == "all":
            self.mut_node_index_sets = pp.mutation_node_index_sets
            self.node_count = pp.node_count - 1
            self.node_indices = tuple(range(1, self.node_count + 1))
        else:
            raise ValueError("Edges must be 'all' or 'internal'.")

        the_sets = lambda: enumerate(self.mut_node_index_sets)
        self.node_index_to_mutation_set_indices = {
            n: tuple(i for i, nodes in the_sets() if n in nodes)
            for n in self.node_indices
        }
        if shuffle_indices:
            # Permute the order of node indices.
            rng = np.random.default_rng(random_seed)
            self.node_indices = tuple(rng.permutation(self.node_indices))
            # Permute each tuple in self.node_index_to_mutation_set_indices.
            for key, value in self.node_index_to_mutation_set_indices.items():
                value = tuple(rng.permutation(value))
                self.node_index_to_mutation_set_indices[key] = value
        self.node_index_to_count_index = {n: i for i, n in enumerate(self.node_indices)}

        self.mut_selections = []
        self.mut_counts = [0] * self.node_count
        self.targeted_node_indices = []

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def alter_mutation_counts(self, mut_set_index, i):
        mut_set = self.mut_node_index_sets[mut_set_index]
        for n in mut_set:
            self.mut_counts[self.node_index_to_count_index[n]] += i

    def increment_mutation_counts(self, mut_set_index):
        self.alter_mutation_counts(mut_set_index, 1)

    def decrement_mutation_counts(self, mut_set_index):
        self.alter_mutation_counts(mut_set_index, -1)

    def any_previous_mut_set_superfluous(self, new_mut_set_index):
        """
        Returns the truth value for the mutation node index set specified by
        new_mut_set_index to cause a previously selected mutation set to become
        unnecessary for all nodes to have mutations.
        """
        self.increment_mutation_counts(new_mut_set_index)
        old_not_needed = False
        for old_mut_set_index in self.mut_selections:
            old_mut_set = self.mut_node_index_sets[old_mut_set_index]
            self.decrement_mutation_counts(old_mut_set_index)
            old_not_needed = all(
                self.mut_counts[self.node_index_to_count_index[n]] > 0
                for n in old_mut_set
            )
            self.increment_mutation_counts(old_mut_set_index)
            if old_not_needed:
                break
        self.decrement_mutation_counts(new_mut_set_index)
        return old_not_needed

    def ordering_check(self, new_mut_set_index):
        """
        This method checks if adding the new mutation set index to the current mutation
        set indices would give a list of indices that is larger than before in terms
        of the complicated lexicographical ordering in the documentation.
        """
        new_mut_set = self.mut_node_index_sets[new_mut_set_index]
        earlier_nodes = new_mut_set.intersection(self.targeted_node_indices[:-1])
        old_inner_indices = (
            self.node_index_to_mutation_set_indices[node].index(
                self.mut_selections[self.targeted_node_indices.index(node)]
            )
            for node in earlier_nodes
        )
        new_inner_indices = (
            self.node_index_to_mutation_set_indices[node].index(new_mut_set_index)
            for node in earlier_nodes
        )
        return all(old < new for old, new in zip(old_inner_indices, new_inner_indices))

    def next(self):
        if self.step_forward_on_mutations():
            return sorted(self.mut_selections)
        else:
            raise StopIteration()

    def step_forward_on_mutations(self):
        """
        Sets self.mut_selections and self.mut_counts to the next valid minimal perfect
        cover of the internal nodes, returning True when this is possible and False when
        there are no more.
        """
        there_are_nodes_without_subs = 0 in self.mut_counts
        if there_are_nodes_without_subs:
            while there_are_nodes_without_subs:
                # Get the next node index requiring a mutation
                n_index = self.node_indices[self.mut_counts.index(0)]
                self.targeted_node_indices.append(n_index)

                # Take the first mutation set that won't make an earlier choice
                # superflous and that maintains the lexicographical ordering. Note at
                # least one such choice exists, so we never backtrack at this step.
                for mut_set_index in self.node_index_to_mutation_set_indices[n_index]:
                    if not self.any_previous_mut_set_superfluous(mut_set_index):
                        if self.ordering_check(mut_set_index):
                            break
                self.mut_selections.append(mut_set_index)
                self.increment_mutation_counts(mut_set_index)
                there_are_nodes_without_subs = 0 in self.mut_counts

            # Now self.current_mutation_selection specifies a new minimum cover.
            return True
        else:
            # We need to roll back the last mutation set, then proceed to the next
            # available mutation set or roll further back.
            need_to_rollback_mutation = True
            while need_to_rollback_mutation:
                # Get the last mutation set we added.
                last_mut_set_index = self.mut_selections.pop()
                self.decrement_mutation_counts(last_mut_set_index)
                # Get the node we chose this mutation set for.
                last_node_index = self.targeted_node_indices[-1]

                # Try to find a new mutation for this node.
                muts_for_node = self.node_index_to_mutation_set_indices[last_node_index]
                previous_index = muts_for_node.index(last_mut_set_index)
                for next_mut_set_index in muts_for_node[previous_index + 1 :]:
                    if not self.any_previous_mut_set_superfluous(next_mut_set_index):
                        if self.ordering_check(next_mut_set_index):
                            need_to_rollback_mutation = False
                            break

                if need_to_rollback_mutation:
                    # There are no more mutations for the node, so we go further back.
                    self.targeted_node_indices.pop()
                    if last_node_index == self.node_indices[0]:
                        # We can't go further back, so we're done enumerating.
                        return False
                else:
                    # We found a valid mutation, so add it in.
                    self.mut_selections.append(next_mut_set_index)
                    self.increment_mutation_counts(next_mut_set_index)
                    there_are_nodes_without_subs = 0 in self.mut_counts
                    if there_are_nodes_without_subs:
                        return self.step_forward_on_mutations()
                    else:
                        # We have the next minimum cover.
                        return True
