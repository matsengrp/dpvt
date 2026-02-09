from ete3 import Tree

from dpvt.models import BaselineReversion


def make_tree(newick, seq_dict):
    """Create a tree with sequences assigned to all nodes."""
    tree = Tree(newick, format=8)
    for node in tree.traverse():
        node.sequence = seq_dict[node.name]
    return tree


def get_internal_node_labels(tree, reversion_labels):
    """Extract reversion labels for internal (non-root, non-leaf) nodes."""
    node_to_idx = {node: i for i, node in enumerate(tree.traverse("preorder"))}
    return {
        node.name: reversion_labels[node_to_idx[node]].item()
        for node in tree.traverse("preorder")
        if not node.is_root() and not node.is_leaf()
    }


class TestBaselineReversion:
    """Tests for BaselineReversion.get_reversion_labels_from_tree."""

    def setup_method(self):
        self.model = BaselineReversion()

    def test_no_reversion(self):
        """A path with no repeated states: A->C->G."""
        tree = make_tree(
            "(((l1,l2)n2,l3)n1)root;",
            {"root": "A", "n1": "C", "n2": "G", "l1": "G", "l2": "G", "l3": "G"},
        )
        labels = self.model.get_reversion_labels_from_tree(tree)
        internal = get_internal_node_labels(tree, labels)
        assert internal["n1"] == 0.0  # A->C, no reversion
        assert internal["n2"] == 0.0  # C->G, no reversion

    def test_simple_reversion(self):
        """A->C->A: the C->A directly reverses A->C."""
        tree = make_tree(
            "(((l1,l2)n2,l3)n1)root;",
            {"root": "A", "n1": "C", "n2": "A", "l1": "A", "l2": "A", "l3": "A"},
        )
        labels = self.model.get_reversion_labels_from_tree(tree)
        internal = get_internal_node_labels(tree, labels)
        assert internal["n1"] == 0.0  # A->C, no reversion
        assert internal["n2"] == 1.0  # C->A, A seen before -> reversion

    def test_multi_step_reversion(self):
        """A->C->G->A: the site returns to A after two intermediate mutations."""
        tree = make_tree(
            "((((l1,l2)n3,l3)n2,l4)n1)root;",
            {
                "root": "A", "n1": "C", "n2": "G", "n3": "A",
                "l1": "A", "l2": "A", "l3": "A", "l4": "A",
            },
        )
        labels = self.model.get_reversion_labels_from_tree(tree)
        internal = get_internal_node_labels(tree, labels)
        assert internal["n1"] == 0.0  # A->C, no reversion
        assert internal["n2"] == 0.0  # C->G, no reversion
        assert internal["n3"] == 1.0  # G->A, A seen before -> reversion

    def test_per_site_independence(self):
        """A reversion at site 0 doesn't affect detection at site 1."""
        # Site 0: A->C->A (reversion at n2)
        # Site 1: G->G->T (no reversion at n2, just a new mutation)
        tree = make_tree(
            "(((l1,l2)n2,l3)n1)root;",
            {
                "root": "AG", "n1": "CG", "n2": "AT",
                "l1": "AT", "l2": "AT", "l3": "AT",
            },
        )
        labels = self.model.get_reversion_labels_from_tree(tree)
        internal = get_internal_node_labels(tree, labels)
        # n2 should be marked as reversion due to site 0 (A->C->A)
        assert internal["n2"] == 1.0

        # Now a tree where neither site has a reversion
        # Site 0: A->C->G (no reversion)
        # Site 1: G->T->C (no reversion)
        tree2 = make_tree(
            "(((l1,l2)n2,l3)n1)root;",
            {
                "root": "AG", "n1": "CT", "n2": "GC",
                "l1": "GC", "l2": "GC", "l3": "GC",
            },
        )
        labels2 = self.model.get_reversion_labels_from_tree(tree2)
        internal2 = get_internal_node_labels(tree2, labels2)
        assert internal2["n2"] == 0.0

    def test_multiple_reversions_on_one_path(self):
        """A->C->A->C: two reversions on the same path."""
        tree = make_tree(
            "((((l1,l2)n3,l3)n2,l4)n1)root;",
            {
                "root": "A", "n1": "C", "n2": "A", "n3": "C",
                "l1": "C", "l2": "C", "l3": "C", "l4": "C",
            },
        )
        labels = self.model.get_reversion_labels_from_tree(tree)
        internal = get_internal_node_labels(tree, labels)
        assert internal["n1"] == 0.0  # A->C, no reversion
        assert internal["n2"] == 1.0  # C->A, A seen before -> reversion
        assert internal["n3"] == 1.0  # A->C, C seen before -> reversion
