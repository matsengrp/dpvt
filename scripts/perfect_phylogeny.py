from ete3 import Tree
from itertools import (
    permutations as perms,
    combinations as combs,
    combinations_with_replacement as combs_r,
    product as prod,
)


class perfect_phylogeny:
    """
    A class to determine multiple sequence alignments and mutation histories that form
    a perfect phylogeny for a given topology.

    Attributes:
        leaf_node_indices (list): A list of leaf node indices in the list of nodes.
        leaf_count (int): The number of leaf nodes in the topology.
        nodes (list): A preorder list of the nodes of the topology. The node indices
            used throughout this class are indices into this list.
            ...preorder, so root is at index 0...

        node_count (int): The nunber of nodes in the topology.
        node_index (dict): A dictionary mapping a node of the topology to its index in
            the list of nodes.
        num_states (int): The number of states.
        states (list): The allowed characters at sites.
        tree (ete3.Tree): The topology.


        valid_mutation_node_index_lists (list): A list of lists, where each inner list
            gives the indices of nodes where mutations occur, such that the mutations
            can be chosen to produce a perfect phylogeny

        unlabelled_state_lists: kinda inbetween the previous two...
            each element is a list that marks a node as 0,1,2,...,num states -1
            where that labelleing will givre a perfect phylo
            after mapping 0,1,2,....num states-1
            to some permutation of the allowed state letters.
            indexed in same order as valid_mutation_node_index_lists





        labelled_state_lists = each element is a list of states for the nodes of the tree
            such that sequences give a perfect (but not necessarilly unique taxa) phylogency





    """

    def __init__(self, tree, states=["A", "G", "C", "T"]):
        self.tree = tree.copy()
        self.states = states
        self.num_states = len(self.states)
        self.nodes = list(self.tree.traverse(strategy="preorder"))
        self.node_count = len(self.nodes)
        self.node_index = dict(zip(self.nodes, range(self.node_count)))
        self.leaf_indices = [self.node_index[leaf] for leaf in self.tree.get_leaves()]
        self.leaf_count = len(self.leaf_indices)
        for node in self.nodes:
            node.add_feature("node_index", self.node_index[node])

        # ...explain why starts at 2 instead of 1
        self.state_permutations = {
            r: [(lambda x: p[x]) for p in list(perms(self.states, r))]
            for r in range(2, self.num_states + 1)
        }

        self.temp_perms = {
            r: list(perms(self.states, r)) for r in range(2, self.num_states + 1)
        }

        # apply the i-th permutation selecting r elts to the i
        self.perm_fn = lambda r, i, x: self.temp_perms[r][i][x]
        print("^????")

        for r in range(2, self.num_states + 1):
            print(f"permutations selecting {r} letters:")
            for j in range(len(self.temp_perms[r])):
                # print(self.temp_perms[r][j])
                derp = [f"{i}->{self.perm_fn(r,j,i)}" for i in range(r)]
                print(", ".join(derp))

            print()
        #        exit()
        print("blarg....")

        self.make_mutation_index_lists()
        self.make_unlabelled_state_lists()
        self.make_labelled_state_lists()

    def make_mutation_index_lists(self):
        """Initialize self.valid_mutation_node_index_lists."""
        self.valid_mutation_node_index_lists = [
            mutated_node_indices
            for num_subs in range(1, self.num_states)
            for mutated_node_indices in combs(range(1, self.node_count), num_subs)
        ]
        return None

    def make_unlabelled_state_lists(self):
        """Initialize self.unlabelled_state_lists."""
        self.unlabelled_state_lists = list(
            map(self.make_unlabelled_state_list, self.valid_mutation_node_index_lists)
        )
        return None

    def make_unlabelled_state_list(self, node_index_list):
        # take the letters as 0, 1, 2, (numstates-1) and handle permutations later
        # the root is assigned 0
        state_list = [0] * self.node_count
        for i, j in enumerate(node_index_list, 1):
            state_list[j] = i
        for node, index in self.node_index.items():
            if index not in node_index_list and not node.is_root():
                parent_index = self.node_index[node.up]
                state_list[index] = state_list[parent_index]
        return state_list

    def make_labelled_state_lists(self):
        """Initialize self.labelled_state_lists."""
        # need to clean up because this will have duplicates. Use some
        # similar call to perms instead....
        perm_fns = [lambda x: p[x] for p in perms(self.states)]
        self.labelled_state_lists = [
            list(map(perm, unlabelled_states))
            for unlabelled_states in self.unlabelled_state_lists
            for perm in perm_fns
        ]
        return None

    def old_node_sequence_from_stuff(
        self,
        node_index,
        indices_into_unlabelled_state_lists,
        perms_to_apply_to_state_lists,
    ):
        return "".join(
            (
                perms_to_apply_to_state_lists[i](
                    self.unlabelled_state_lists[j][node_index]
                )
                for i, j in enumerate(indices_into_unlabelled_state_lists)
            )
        )

    def node_sequence_from_stuff(
        self,
        node_index,
        indices_into_unlabelled_state_lists,
        small_perms_index_list,
    ):
        """
        the sequence for the node at the node_index,
        using the state lists specified,
        applying to the i-th state the permutation at small_perms_index_list[i]
        of the correct size permutations.
        """

        r = lambda i: len(self.valid_mutation_node_index_lists[i]) + 1
        zipped = zip(small_perms_index_list, indices_into_unlabelled_state_lists)

        return "".join(
            (
                self.perm_fn(r(j), i, self.unlabelled_state_lists[j][node_index])
                for i, j in zipped
            )
        )

    def node_sub_from_stuff(
        self, node_index, indices_into_unlabelled_state_lists, small_perms_index_list
    ):
        # need to work out nicer format for list of subs...
        r = lambda i: len(self.valid_mutation_node_index_lists[i]) + 1
        subs = []
        if node_index != 0:
            parent_node_index = self.nodes[node_index].up.node_index
            zipped = zip(indices_into_unlabelled_state_lists, small_perms_index_list)
            for i, (state_list_index, perm_index) in enumerate(zipped):
                parent_at_site = self.unlabelled_state_lists[state_list_index][
                    parent_node_index
                ]
                node_at_site = self.unlabelled_state_lists[state_list_index][node_index]
                if parent_at_site != node_at_site:
                    parent = self.perm_fn(
                        r(state_list_index), perm_index, parent_at_site
                    )
                    child = self.perm_fn(r(state_list_index), perm_index, node_at_site)
                    subs.append(f"{i}_{parent}->{child}")

        derp = "{" + "|".join(subs) + "}"
        return derp

    # later can add arsg for unqiue leaves and mybe distinct sites, then return a tuple
    # (tree, bool for tree passing whatever checks based on args)
    # a little wasteful, but not awful
    def make_certain_tree(
        self,
        label_seq,
        label_sub,
        indices_into_unlabelled_state_lists,
        small_perms_index_list,
    ):
        tree = self.tree.copy()
        for node in tree.traverse(strategy="preorder"):
            node_index = node.node_index
            if label_seq:
                sequence = self.node_sequence_from_stuff(
                    node_index,
                    indices_into_unlabelled_state_lists,
                    small_perms_index_list,
                )
                node.add_feature("sequence", sequence)
            if label_sub:
                sub = self.node_sub_from_stuff(
                    node_index,
                    indices_into_unlabelled_state_lists,
                    small_perms_index_list,
                )
                node.add_feature("sub", sub)

        return (tree, True)

    def index_perm_list_thing(self, unlabelled_state_list_indices):
        r = lambda i: len(self.valid_mutation_node_index_lists[i]) + 1
        return [
            list(range(len(self.state_permutations[r(i)])))
            for i in unlabelled_state_list_indices
        ]

    def bunch_of_trees(
        self,
        label_seq=True,
        label_sub=True,
        unique_leaf_seq=False,
        max_sites=1,
        distinct_sites=False,  # includes full mutation history, not just msa
    ):
        """...only one tree per msa on all nodes (not just leaves), with msa's the same after site rearrangement"""

        n = len(self.unlabelled_state_lists)
        trees = []
        for m in range(1, max_sites + 1):
            for unlabelled_state_list_indices in combs_r(range(n), m):
                big_perms_index_list = self.index_perm_list_thing(
                    unlabelled_state_list_indices
                )
                for small_perms_index_list in prod(*big_perms_index_list):
                    # will apply perm from small_perms_index_list[i] to unlabelled_state_list_indices[i]

                    tree, is_valid = self.make_certain_tree(
                        label_seq,
                        label_sub,
                        unlabelled_state_list_indices,
                        small_perms_index_list,
                    )
                    if is_valid:
                        trees.append(tree)

        return trees

    # version before passing indices for permutations
    def old_bunch_of_trees(
        self,
        label_seq=True,
        label_sub=True,
        unique_leaf_seq=False,
        max_sites=1,
        distinct_sites=False,  # includes full mutation history, not just msa
    ):
        """...only one tree per msa on all nodes (not just leaves), with msa's the same after site rearrangement"""

        # probably do want to make this an instanvce variable, then can pass indices around

        n = len(self.unlabelled_state_lists)
        trees = []
        for m in range(1, max_sites + 1):
            for unlabelled_state_list_indices in combs_r(range(n), m):
                big_perms_list = self.perm_list_thing(unlabelled_state_list_indices)
                for small_perms_list in prod(*big_perms_list):
                    tree, is_valid = self.make_certain_tree(
                        label_seq,
                        label_sub,
                        unlabelled_state_list_indices,
                        small_perms_list,
                    )
                    if is_valid:
                        trees.append(tree)

        return trees

    def mutation_histories_with_unique_leaf_sequences(self, max_sites=10):
        """
        Returns a generator for the list of combinations (not permutations) of indices
        into self.labelled_state_lists. The states at each combination of indices
        provide a full mutation history, with unique leaf sequences, along the topology
        that is a perfect phylogeny.
        """
        n = len(self.labelled_state_lists)
        return (
            state_list_indices
            for r in range(1, max_sites + 1)
            for state_list_indices in combs(range(n), r)
            if self.state_lists_give_unique_leaves(state_list_indices)
        )

    def node_seq_from_state_list_indices(self, node_index, indices):
        """Returns the sequence for the node using the given state list indices."""
        return "".join((self.labelled_state_lists[i][node_index] for i in indices))

    def state_lists_give_unique_leaves(self, indices):
        """
        Returns the truth value of the leaf node sequences being distinct, when taking
        the sequences from the state lists at the given indices.
        """
        seq = lambda x: self.node_seq_from_state_list_indices(x, indices)
        return self.leaf_count == len(set(map(seq, self.leaf_indices)))

    def make_tree_from_state_lists(self, indices):
        """
        Returns a new tree instance whose nodes have the feature 'sequence' recording
        the full sequence at each node, with the sequence determined by the state lists
        at the given indices.
        """
        tree = self.tree.copy()
        for node in tree.traverse(strategy="preorder"):
            node_index = node.node_index
            sequence = self.node_seq_from_state_list_indices(node_index, indices)
            node.add_feature("sequence", sequence)
        return tree

    # maybe code something up that allows adding sites to a tree?

    def make_just_subs_trees(self, node_index_list, all_perms=True):
        # root is assigned 0
        which_nodes = [0]
        which_nodes.extend(node_index_list)
        which_letters = list(range(node_index_list + 1))
        if all_perms:
            r = len(node_index_list) + 1
            perm_fns = [lambda x: p[x] for p in perms(self.states, r)]
        else:
            perm_fns = [lambda x: self.states[x]]

        trees = [self.tree.copy() for _ in range(len(perm_fns))]
        for tree, perm in zip(trees, perm_fns):
            letters = map(perm, which_letters)
            for node in tree.traverse(strategy="preorder"):
                node_index = node.node_index
                if node_index in which_nodes:
                    node.add_feature("sub", next(letters))

        return trees

    def make_tree_just_subs(self, indices):
        """
        Returns a new tree instance whose nodes have the feature 'subs' recording
        the site mutations from the parent node.
        """
        pass


if __name__ == "__main__":
    # newick = "(((((a,b),c),d),e),f);"
    newick = "(a,(b,c));"
    tree = Tree(newick)

    thing = perfect_phylogeny(tree)

    trees = thing.bunch_of_trees(max_sites=2)
    for tree in trees:
        print(tree.write(features=["sequence", "sub"], format=9))
        print(f"Ancestral Sequence: {tree.sequence}")
        print()
