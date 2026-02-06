import warnings
import torch
from torch import nn
import torch.nn.functional as F
from torchmetrics import AUROC
from torchmetrics.classification import BinaryROC, BinaryAccuracy, BinaryConfusionMatrix
from torch.nn.utils.rnn import pad_sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lightning as L
from ete3 import Tree

# DNA states
STATES = ["A", "G", "C", "T"]
STATE_TO_IDX = {"A": 0, "G": 1, "C": 2, "T": 3}
n_states = len(STATES)

# neural network parameters
learning_rate = 0.01

# traverse stack parameters
d_hidden_traverse = 32

# site-aggregate transformer parameters
nhead = 2
# coming from concatenating the traversal output-feature in two directions
# across edge
dim_feedforward = 8
layer_count = 4

# for matrix multiplications with Tensor Cores
torch.set_float32_matmul_precision("high")


class TraverseNN(L.LightningModule):
    """
    A pytorch module which takes a list of ete3.Trees as input and outputs a
    list of vectors of 0's and 1's to indicate whether each edge of each input
    tree is maximally parsimonious or not, respectively, for the sequences
    assigned to the leaf nodes.

    The forward function applies two traversals to the input tree, first
    root-ward and then leaf-ward. The traversals combine data across the tree
    for each site, keeping data from separate sites separate. Then, a
    transformer encoder is used to combine data across separate sites, at each
    node. The transformer encoder outputs (n_sites)- many vectors but we only
    keep the data in the first vector. Finally, this encoder output vector is
    passed through a linear classifier and a sigmoid to get a prediction.

    Attributes:
        traverse_stack: MLP with single hidden layer, used to summarize mutation
        data
            below a given node at a given site, by combining data from its two
            children
        encoder_layer: encoder: transformer encoder used to summarize mutation
        data across all sizes,
            at a given node
        classifier: linear layer to produce prediction logit, for whether each
        edge is
            present in a maximum parsimony tree
    """

    def __init__(self, learning_rate=0.01, feature_length=32, dim_mlp_layers=32):
        super().__init__()
        self.lr = learning_rate
        self.feature_length = feature_length
        self.dim_mlp_layers = dim_mlp_layers
        self.traverse_stack = nn.Sequential(
            nn.Linear(2 * n_states + 2 * feature_length, dim_mlp_layers),
            nn.ReLU(),
            nn.Linear(dim_mlp_layers, feature_length),
        )
        self.d_model = 2 * feature_length
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, layer_count)
        self.classifier = nn.Linear(self.d_model, 1)
        self.roc_metric = BinaryROC()
        self.auroc_metric = AUROC(task="binary")
        self.accuracy_metric = BinaryAccuracy()
        # Temporary storage for probabilities and targets
        self.test_probs = []
        self.test_targets = []

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def masked_bce_loss(self, pred, target, mask):
        loss = F.binary_cross_entropy_with_logits(
            pred, target.unsqueeze(-1), reduction="none"
        )
        masked_loss = loss * mask.unsqueeze(-1)  # element-wise multiplication
        return masked_loss.mean()

    def _process_batch(self, batch):
        """
        Process a batch of data and return predictions, labels, and mask.

        Handles both Tree dataset format (trees, labels, mask) and
        traversal dataset format (traversal, mutations, labels, mask).

        Returns:
            tuple: (pred, labels, mask) where pred contains logits
        """
        if isinstance(batch[0][0], Tree):
            trees, labels, mask = batch
            max_seq_length = max(len(tree.sequence) for tree in trees)
            predictions = [self.forward_on_tree(item, max_seq_length) for item in trees]
            # Pad predictions if trees have varying number of leaves
            max_length = labels.size(1)
            padded_predictions = [
                F.pad(p, (0, 0, 0, max_length - p.size(0))) for p in predictions
            ]
            pred = torch.stack(padded_predictions)
        else:
            traversal, mutations, labels, mask = batch
            predictions = [
                self.forward_on_traversal(t, m) for (t, m) in zip(traversal, mutations)
            ]
            pred = torch.stack(predictions)
        return pred, labels, mask

    def training_step(self, train_batch, batch_idx):
        pred, labels, mask = self._process_batch(train_batch)
        prob_predictions = F.sigmoid(pred)
        # labels < 0.5 selects edges IN MP tree (label=0), labels >= 0.5 selects edges NOT in MP tree (label=1)
        mp_edge_predictions = prob_predictions[labels < 0.5]
        non_mp_edge_predictions = prob_predictions[labels >= 0.5]
        self.log("mp_edge_pred_avg", torch.mean(mp_edge_predictions), prog_bar=True)
        self.log(
            "non_mp_edge_pred_avg", torch.mean(non_mp_edge_predictions), prog_bar=True
        )
        loss = self.masked_bce_loss(pred, labels, mask)
        self.log("train_loss", loss.item(), on_epoch=True, batch_size=len(labels))
        return loss

    def validation_step(self, val_batch, batch_idx):
        pred, labels, mask = self._process_batch(val_batch)
        loss = self.masked_bce_loss(pred, labels, mask)
        self.log(
            "val_loss",
            loss.item(),
            on_epoch=True,
            prog_bar=True,
            batch_size=len(labels),
        )
        # Return the batch loss so it can be accumulated later
        return loss

    def test_step(self, test_batch):
        pred, labels, mask = self._process_batch(test_batch)
        # only get unmasked output
        masked_pred = pred[mask]
        masked_labels = labels[mask].unsqueeze(-1).int()
        self.test_probs.append(masked_pred)
        self.test_targets.append(masked_labels)
        if torch.numel(masked_labels) > 0:  # Check if there are any unmasked elements
            self.auroc_metric(masked_pred, masked_labels)
            probs = torch.sigmoid(masked_pred)
            self.accuracy_metric(probs, masked_labels)
            loss = self.masked_bce_loss(pred, labels, mask)
            self.log("test_loss", loss.item(), batch_size=len(labels))
        else:
            warnings.warn("Your test data is very small, or there is a bug")
        return {}

    def on_validation_epoch_end(self):
        # average validation loss over epoch and plot
        outputs = self.trainer.callback_metrics
        avg_val_loss = outputs["val_loss"]
        # self.log('val_loss', avg_val_loss, on_epoch=True, batch_size = len(),
        # prog_bar=True)
        self.logger.experiment.add_scalars(
            "loss", {"valid": avg_val_loss.item()}, self.current_epoch
        )

    def on_train_epoch_end(self):
        # average training loss over epoch and plot
        outputs = self.trainer.callback_metrics
        avg_train_loss = outputs["train_loss"]
        if avg_train_loss is not None:
            self.logger.experiment.add_scalars(
                "loss", {"train": avg_train_loss.item()}, self.current_epoch
            )

    def on_test_epoch_end(self):
        """
        Summarize testing statistics (AUROC and ROC) at end of testing and adds
        them to logging. The ROC curve is added as figure to self.logger.
        """
        logits = torch.cat(self.test_probs, dim=0)
        targets = torch.cat(self.test_targets, dim=0)
        probs = torch.sigmoid(logits)

        auroc = self.auroc_metric.compute()
        self.log("test_auroc", auroc.item(), on_step=False, on_epoch=True)
        accuracy = self.accuracy_metric.compute()
        self.log("test_accuracy", accuracy.item(), on_step=False, on_epoch=True)

        fpr, tpr, thresholds = self.roc_metric(probs, targets)
        fpr = fpr.cpu()
        tpr = tpr.cpu()

        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUROC: {auroc:.2f}")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")

        if self.logger:
            self.logger.experiment.add_figure("ROC Curve", fig, self.current_epoch)

        plt.close(fig)

        # Clear the stored probabilities and targets
        self.test_probs.clear()
        self.test_targets.clear()
        self.roc_metric.reset()
        self.accuracy_metric.reset()

    def forward(self, input, optimized=False):
        """
        Takes an iterable of ete3.Tree as input and outputs a tensor of 0's and
        1's to indicate whether each input tree is maximally parsimonious or
        not, respectively, for the sequences assigned to the leaf nodes. Args:
            input (list of Trees): has attribute to_parent["edge_mutation"] on
            each node,
                which is a torch tensor that encodes the mutation between the
                node and its parent, e.g. A -> G is encoded by [-1, 1, 0, 0]
            optimized (boolean): if True, runs more efficiently by skipping type
            check
                that allows ete.Tree input
        """
        if not optimized:
            if type(input) == Tree:
                max_seq_length = len(input.sequence)
                logit = self.forward_on_tree(input, max_seq_length)
                return F.sigmoid(logit)
        # assume input is a list (or iterable) of trees
        if type(input[0]) == Tree:
            max_seq_length = max([len(tree.sequence) for tree in input])
            logits = torch.stack(
                [self.forward_on_tree(item, max_seq_length) for item in input]
            )
        else:
            logits = torch.stack([self.forward_on_traversal(item) for item in input])
        return F.sigmoid(logits)

    def forward_on_tree(self, tree: Tree, max_seq_length):
        """
        Takes an ete3.Tree as input and outputs a list of 0's and 1's to
        indicate whether each edge on the input tree is present in a maximally
        parsimonious tree or not, respectively, for the sequences assigned to
        the leaf nodes. Args:
            tree (ete3 Tree): each node has a sequence attribute, which is a
            string
                consisting of the characters A, G, C, T
        """
        self.assign_mutation_vectors(tree)
        self.compute_features_via_traversal(tree, max_seq_length)
        encoder_output = self.site_aggregate(tree)
        # encoder_output dim = (n_nodes, 1, 8)
        summarized_features = encoder_output.mean(dim=1)
        logit = self.classifier(summarized_features)
        return logit

    def traverse_node_aggregate(
        self,
        first_node_mutations,
        first_node_feature,
        second_node_mutations,
        second_node_feature,
        output_feature,
    ):
        """
        Takes in feature vectors from two neighbor-nodes of a given node, and
        outputs the feature vector for that node. The two neighbor-nodes can be
        either:
            - two children of a node, during root-ward traversal, or
            - one parent and one sister of a node, during leaf-ward traversal.
        """
        combined_data = torch.cat(
            (
                first_node_mutations,
                first_node_feature,
                second_node_mutations,
                second_node_feature,
            ),
            dim=0,
        )
        output = self.traverse_stack(combined_data)
        output_feature.copy_(output)

    def forward_on_traversal(self, traversal, mutations):
        """summarized_features = encoder_output.mean(dim=0)
        Compute features from traversal data structure, given one tree/traversal
        and corresponding mutations (for all sites), then aggregates and
        classifies.
        """
        learned_features = self.traversal_on_traversal(traversal, mutations)
        attention_masks = (
            (learned_features == 0).any(dim=2).to(torch.bool).transpose(0, 1)
        )
        encoder_output = self.encoder(
            learned_features, src_key_padding_mask=attention_masks
        )
        summarized_features = encoder_output.mean(dim=1)
        logit = self.classifier(summarized_features)
        return logit

    def traversal_on_traversal(self, traversal, mutations):
        """
        Compute features from traversal datastructure, given one tree/traversal
        and corresponding mutations (for all sites).
        """
        max_seq_length = mutations.size(1)
        device = traversal.device

        # Initialize a single preallocated tensor to store features for all nodes
        # Shape: (max_node, 2, max_seq_length, feature_length)
        max_node = len(mutations)
        node_features = torch.zeros(
            max_node, 2, max_seq_length, self.feature_length, device=device
        )

        for i, direction in enumerate(traversal):
            for node_list in direction:
                # first two nodes in list have feature already, we learn feature
                # for third node in list
                current_node = int(node_list[2])
                # adjacent nodes (either children or sibling and parent)
                adj_node1 = int(node_list[0])
                adj_node2 = int(node_list[1])
                if current_node == adj_node1 == adj_node2:
                    # stop if we are in padded part of traversal representation
                    break
                if i == 0:
                    # in upward traversal, multiply mutation encodings of both children by -1
                    mutation1 = -1 * mutations[adj_node1]
                    mutation2 = -1 * mutations[adj_node2]
                else:
                    # in downward traversal, only multiply mutation encoding of sibling by -1
                    mutation1 = -1 * mutations[adj_node1]
                    mutation2 = mutations[adj_node2]
                # Compute features for the current node
                combined_data = torch.cat(
                    (
                        mutation1,
                        node_features[adj_node1, i],
                        mutation2,
                        node_features[adj_node2, i],
                    ),
                    dim=1,
                )

                # Update just the features we need
                node_features[current_node, i] = self.traverse_stack(combined_data)
        # Concatenate features -> (max_node, max_seq_length, 2 * feature_length)
        learned_features = node_features.permute(0, 2, 1, 3).reshape(
            max_node, max_seq_length, 2 * self.feature_length
        )
        return learned_features

    def compute_features_via_traversal(
        self,
        tree: Tree,
        max_seq_length,
        feature_name="edge_mutation",
    ):
        """
        Takes an ete3.Tree as input and assigns a feature to each node which is
        a tensor of dimension (n_edges, n_sites, n_states=4). At each edge and
        each site, the tensor encodes a summary of the mutations that occurred
        on the subtree on either side of the specified edge, at the specified
        site. Args:
            tree (ete3 Tree): each node has a torch tensor attribute
                to_parent["edge_mutation"] that encodes the mutation between the
                node and its parent, e.g. A -> G is encoded by [-1, 1, 0, 0]
            seq_length (int): length of input sequences feature_name (string):
            name of feature assigned to nodes of the tree that
                are used for encoding
        """
        # root-ward traversal
        seq_length = len(tree.sequence)
        for node in tree.traverse(strategy="postorder"):
            if node.is_leaf() or node.is_root() or node.up.is_root():
                node.to_parent["clade_mutation_feature"] = torch.zeros(
                    (max_seq_length, self.feature_length)
                )
            else:
                feature = torch.zeros((max_seq_length, self.feature_length))
                try:
                    child1, child2 = node.children
                except ValueError:
                    raise ValueError(
                        f"Input tree must be bifurcating, but node has"
                        "{len(node.children)} children"
                    )
                for i in range(seq_length):
                    self.node_aggregate(
                        child1.to_parent,
                        child2.to_parent,
                        feature_name,
                        feature[i],
                        site_idx=i,
                    )
                node.to_parent["clade_mutation_feature"] = feature
        # leaf-ward traversal
        for node in tree.traverse(strategy="preorder"):
            feature = torch.zeros((max_seq_length, self.feature_length))
            if node.is_root() or node.is_leaf():
                node.from_parent["clade_mutation_feature"] = feature
            elif node.up.is_root():
                assert (
                    len(node.up.children) == 1
                ), "Error: root of tree should have single child"
                node.from_parent["clade_mutation_feature"] = feature
            else:
                parent = node.up
                # `node` should have a single sister node
                assert len(node.get_sisters()) == 1
                sister = node.get_sisters()[0]
                for i in range(seq_length):
                    self.node_aggregate(
                        parent.from_parent,
                        sister.to_parent,
                        feature_name,
                        feature[i],
                        site_idx=i,
                    )
                node.from_parent["clade_mutation_feature"] = feature
        return tree

    def site_aggregate(self, tree):
        """
        Takes a tensor encoding site-wise mutations on subclades of a tree and
        aggregates its n_sites using a Transformer
        """
        learned_features = torch.stack(
            [
                torch.cat(
                    (
                        node.to_parent["clade_mutation_feature"],
                        node.from_parent["clade_mutation_feature"],
                    ),
                    dim=1,
                )
                for node in tree.traverse(strategy="preorder")
            ]
        )
        # learned_features dim = (n_nodes, n_sites, d_model=8)
        learned_features.transpose(0, 1)
        attention_masks = (learned_features == 0).any(dim=2).transpose(0, 1)
        encoder_output = self.encoder(
            learned_features, src_key_padding_mask=attention_masks
        )
        # out dim = (n_nodes, n_sites, d_model=8)
        return encoder_output

    def node_aggregate(
        self,
        first_node_dict,
        second_node_dict,
        feature_name,
        output_feature,
        site_idx,
        symmetrize=False,
    ):
        """
        Takes in dictionaries of feature vectors from two neighbor-nodes of a
        given node, and outputs the `clade_mutation_feature` vector for that
        node. The two neighbor-nodes can be either:
            - two children of a node, during root-ward traversal, or
            - one parent and one sister of a node, during leaf-ward traversal.
        """
        combined_data = torch.cat(
            (
                first_node_dict[feature_name][site_idx],
                first_node_dict["clade_mutation_feature"][site_idx],
                second_node_dict[feature_name][site_idx],
                second_node_dict["clade_mutation_feature"][site_idx],
            ),
            dim=0,
        )
        output = self.traverse_stack(combined_data)
        if symmetrize:
            output += self.traverse_stack(
                torch.cat(
                    (
                        first_node_dict[feature_name][site_idx],
                        first_node_dict["clade_mutation_feature"][site_idx],
                        second_node_dict[feature_name][site_idx],
                        second_node_dict["clade_mutation_feature"][site_idx],
                    )
                )
            )
        output_feature.copy_(output)

    @staticmethod
    def assign_mutation_vectors(tree):
        """
        Modifies input tree by adding a `to_parent` and `from_parent` dict
        attributes. `to_parent["edge_mutation"]` is a size (n_sites, n_states=4)
        torch.tensor which records the mutation from the node's parent to the
        (child) node, e.g., a mutation `A -> T` is encoded as [...,[-1, 0, 0,
        1],...] Args:
            tree (ete3 Tree): has sequence attribute on each node
        Returns: None
        """
        n_sites = len(tree.sequence)
        for node in tree.traverse():
            for i in range(n_sites):
                mut_vec = [0.0, 0.0, 0.0, 0.0]
                if node.up is None:  # node is root
                    pass
                else:  # non-root node
                    n_seq = node.sequence[i]
                    p_seq = node.up.sequence[i]
                    try:
                        mut_vec[STATE_TO_IDX[n_seq]] += 1
                        mut_vec[STATE_TO_IDX[p_seq]] -= 1
                    except KeyError:
                        raise ValueError(f"Each node sequence must be in {STATES}")
                new_row = torch.tensor(mut_vec).unsqueeze(0)
                if i == 0:
                    node.add_feature("to_parent", {"edge_mutation": new_row})
                    node.add_feature("from_parent", {"edge_mutation": -new_row})
                else:
                    node.to_parent["edge_mutation"] = torch.cat(
                        (node.to_parent["edge_mutation"], new_row)
                    )
                    node.from_parent["edge_mutation"] = torch.cat(
                        (node.from_parent["edge_mutation"], -new_row)
                    )
        return None


class TraverseMaxPooling(TraverseNN):
    """
    Pytorch module inherited from TraverseNN, which replaces the site
    aggregation by taking the maximum feature of the output of the MLP for
    classification (maximum over all sites)
    """

    def site_aggregate(self, tree):
        """
        Takes an encoding of the root sequence of a tree and aggregates its
        n_sites by choosing the max entry of each feature position over all
        sites
        """
        input_features = torch.stack(
            [
                torch.cat(
                    (
                        node.to_parent["clade_mutation_feature"],
                        node.from_parent["clade_mutation_feature"],
                    ),
                    dim=1,
                )
                for node in tree.traverse(strategy="preorder")
            ]
        )
        max_values, _ = torch.max(input_features, dim=1, keepdim=True)
        return max_values

    def forward_on_traversal(self, traversal, mutations):
        """
        Compute features from traversal datastructure, given one tree/traversal
        and corresponding mutations (for all sites), then aggregates and
        classifies.
        """
        learned_features = self.traversal_on_traversal(traversal, mutations)
        output, _ = torch.max(learned_features, dim=1, keepdim=True)
        logit = self.classifier(output[:, 0])
        return logit


class TraverseAvgPooling(TraverseNN):
    """
    Pytorch module inherited from TraverseNN, which replaces the site
    aggregation by taking the average of features over all sites
    """

    def __init__(self, learning_rate=0.01, feature_length=32, dim_mlp_layers=32):
        super().__init__()
        self.lr = learning_rate
        self.feature_length = feature_length
        self.dim_mlp_layers = dim_mlp_layers
        self.d_model = 2 * feature_length
        self.traverse_stack = nn.Sequential(
            nn.Linear(2 * n_states + 2 * feature_length, dim_mlp_layers),
            nn.ReLU(),
            nn.Linear(dim_mlp_layers, feature_length),
        )
        self.classifier = nn.Linear(self.d_model, 1)
        self.roc_metric = BinaryROC()
        self.auroc_metric = AUROC(task="binary")
        self.accuracy_metric = BinaryAccuracy()
        # Temporary storage for probabilities and targets
        self.test_probs = []
        self.test_targets = []

    def site_aggregate(self, tree):
        """
        Takes an encoding of the root sequence of a tree and aggregates its
        n_sites by choosing the feature of the site with max sum over all
        features entries
        """
        input_features = torch.stack(
            [
                torch.cat(
                    (
                        node.to_parent["clade_mutation_feature"],
                        node.from_parent["clade_mutation_feature"],
                    ),
                    dim=1,
                )
                for node in tree.traverse(strategy="preorder")
            ]
        )
        col_sums = input_features.mean(dim=1, keepdim=True)
        return col_sums

    def forward_on_traversal(self, traversal, mutations):
        """
        Compute features from traversal data structure, given one tree/traversal
        and corresponding mutations (for all sites), then aggregate and
        classify.
        """
        learned_features = self.traversal_on_traversal(traversal, mutations)
        output = learned_features.mean(dim=1, keepdim=True)
        logit = self.classifier(output[:, 0])
        return logit


class BaselineReversion(L.LightningModule):
    """
    Baseline model we compare our deep learning models to. This model detects
    whether there is a reversion at any site on an edge and if so, labels it as
    non-MP (computed in get_reversion_labels_from_tree). A reversion is detected
    when a site returns to any previously-seen state along the root-to-node path,
    including through multiple intermediate mutations (e.g. A->C->G->A). This
    model requires the input to be in the format of a TreeDataset.
    """

    def __init__(self):
        super().__init__()
        # No learnable parameters needed
        self.roc_metric = BinaryROC()
        self.auroc_metric = AUROC(task="binary")
        self.accuracy_metric = BinaryAccuracy()
        self.confusion_matrix_metric = BinaryConfusionMatrix()
        # Temporary storage for probabilities and targets
        self.test_probs = []
        self.test_targets = []

    def get_reversion_labels_from_tree(self, tree):
        """
        Take in a tree object and create a list with labels 0/1 containing for
        each node whether there is a reversion on the edge above the node.
        A reversion is detected when a mutation changes a site to a state that
        has already been seen earlier on the path from root to that node.
        """
        n_sites = len(tree.sequence)
        # Track the set of states seen at each site along the root-to-node path
        node_state_history = {
            node: {site_idx: set() for site_idx in range(n_sites)}
            for node in tree.traverse()
        }

        # Initialise root: record the root's state at each site
        root = tree.get_tree_root()
        for site_idx in range(n_sites):
            node_state_history[root][site_idx].add(root.sequence[site_idx])

        # Result tensor storing reversion status for each node
        reversion_labels = torch.zeros(len(list(tree.traverse())), dtype=torch.float32)

        # Map nodes to indices to keep consistent indexing
        node_to_idx = {node: i for i, node in enumerate(tree.traverse("preorder"))}

        # Process from root to leaves (preorder traversal)
        for node in tree.traverse("preorder"):
            if node.is_root() or node.is_leaf():
                # root and leaf nodes will be labelled as MP edges
                continue

            parent = node.up
            # For each site, track mutations
            for site_idx in range(n_sites):
                # Copy parent's state history
                node_state_history[node][site_idx] = node_state_history[parent][
                    site_idx
                ].copy()

                n_seq = node.sequence[site_idx]
                p_seq = parent.sequence[site_idx]

                if n_seq != p_seq:
                    # Check if the new state was seen before on this path
                    if n_seq in node_state_history[node][site_idx]:
                        reversion_labels[node_to_idx[node]] = 1
                    # Record the new state
                    node_state_history[node][site_idx].add(n_seq)

        return reversion_labels

    def forward(self, batch):
        """Apply the reversion detection to the input batch"""
        # Input is a list of trees
        reversion_labels = [self.get_reversion_labels_from_tree(tree) for tree in batch]
        return reversion_labels

    def test_step(self, test_batch, batch_idx):
        """Test step that handles both tree datasets and tensor datasets"""
        if type(test_batch[0][0]) == Tree:
            # Input is tree dataset
            trees, labels, mask = test_batch
            max_seq_length = max([len(tree.sequence) for tree in trees])
            predictions = self.forward(trees)
            max_num_leaves = labels.size(1)  # labels are already padded
            predictions = pad_sequence(predictions, batch_first=True, padding_value=0)

        else:
            raise TypeError(
                f"Expected input to be of type ete3.Tree, "
                f"but got {type(test_batch[0][0]).__name__}. "
                f"This model only supports TreeDataset format."
            )

        # Apply mask to focus on the edges we care about
        masked_pred = predictions[mask]
        masked_labels = labels[mask].int()

        # Store predictions for ROC computation
        self.test_probs.append(masked_pred)
        self.test_targets.append(masked_labels)

        # Set up AUROC and accuracy metrics
        if torch.numel(masked_labels) > 0:
            self.auroc_metric(masked_pred, masked_labels)
            # Convert binary predictions (0/1) to probability format (0.0/1.0) for accuracy calculation
            binary_probs = masked_pred.float()  # Convert to float if not already
            self.accuracy_metric(binary_probs, masked_labels)
            self.confusion_matrix_metric(binary_probs, masked_labels)

        return {"predictions": predictions, "labels": labels, "mask": mask}

    # Required by PyTorch Lightning but won't be used
    def configure_optimizers(self):
        return None

    # Override training_step to comply with Lightning's requirements
    def training_step(self, batch, batch_idx):
        # This will never actually be called
        return None

    def on_test_epoch_end(self):
        """Compute final metrics at end of testing"""
        if not self.test_probs:
            return

        preds = torch.cat(self.test_probs, dim=0)
        targets = torch.cat(self.test_targets, dim=0)
        # Compute and save metrics
        auroc = self.auroc_metric.compute()
        self.log("test_auroc", auroc.item(), on_step=False, on_epoch=True)
        accuracy = self.accuracy_metric.compute()
        self.log("test_accuracy", accuracy.item(), on_step=False, on_epoch=True)

        # Compute confusion matrix and extract TP, FP, TN, FN
        confusion_matrix = self.confusion_matrix_metric.compute()
        # confusion_matrix has shape [[TN, FP], [FN, TP]]
        tn = confusion_matrix[0, 0].item()
        fp = confusion_matrix[0, 1].item()
        fn = confusion_matrix[1, 0].item()
        tp = confusion_matrix[1, 1].item()

        self.log("test_true_negatives", tn, on_step=False, on_epoch=True)
        self.log("test_false_positives", fp, on_step=False, on_epoch=True)
        self.log("test_false_negatives", fn, on_step=False, on_epoch=True)
        self.log("test_true_positives", tp, on_step=False, on_epoch=True)

        # Ensure preds are in proper probability format for ROC calculation
        if preds.unique().numel() <= 2 and preds.max() <= 1.0:
            # If binary predictions (0/1), use as is
            roc_preds = preds.float()
        else:
            # Otherwise, normalize to 0-1 range
            roc_preds = (preds - preds.min()) / (preds.max() - preds.min())

        fpr, tpr, thresholds = self.roc_metric(roc_preds, targets)

        # Create ROC curve
        fig, ax = plt.subplots()
        ax.plot(fpr.cpu(), tpr.cpu(), label=f"AUROC: {auroc:.2f}")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve - Baseline Reversion")
        ax.legend(loc="lower right")

        if self.logger:
            self.logger.experiment.add_figure("ROC Curve", fig, 0)

        plt.close(fig)

        # Clear stored data
        self.test_probs.clear()
        self.test_targets.clear()
        self.roc_metric.reset()
        self.accuracy_metric.reset()
        self.confusion_matrix_metric.reset()
