import torch
from torch import nn
import torch.nn.functional as F
from torchmetrics import AUROC
from torchmetrics.classification import BinaryROC
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lightning as L
from ete3 import Tree

# transformer parameters
nhead = 2
d_model = (
    4  # size of embedding that we feed into transformer, i.e. length of mutation vector
)
dim_feedforward = 8
layer_count = 4
learning_rate = 0.01


class TraverseNN(L.LightningModule):
    """
    A pytorch module which takes a list of ete3.Trees as input and outputs 0's and 1's
    to indicate whether each input tree is maximally parsimonious or not, respectively,
    for the sequences assigned to the leaf nodes.

    The forward function applies two traversals to the input tree, first root-ward and
    then leaf-ward.

    For now, we only implement the root-ward traversal.

    Attributes:
        up_traverse_stack: NN with single hidden layer, used to summarize mutation data
            below a given node at a given site, by combining data from its two children
        encoder_layer:
        encoder: transformer encoder used to summarize mutation data across all sizes,
            at a given node
        final_on_site:
        final_across_sites:
    """

    def __init__(self, learning_rate):
        super().__init__()
        # learning rate
        self.lr = learning_rate
        self.up_traverse_stack = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
        )
        # self.down_traverse_stack = nn.Sequential(
        #     nn.Linear(16, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 4),
        # )
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, layer_count)
        self.classifier = nn.Linear(d_model, 1)
        self.roc_metric = BinaryROC()
        self.auroc_metric = AUROC(task="binary")
        # Temporary storage for probabilities and targets
        self.test_probs = []
        self.test_targets = []

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        xb, yb = train_batch
        pred = self(xb)
        loss = F.binary_cross_entropy(pred, yb.unsqueeze(1))
        self.log("train_loss", loss, batch_size=len(xb), on_epoch=True)
        # log predictions on positive- and negative-datapoints, and show data in console
        # progress bar
        prob_predictions = F.sigmoid(pred)
        pos_predictions = prob_predictions[yb < 0.5]
        neg_predictions = prob_predictions[yb >= 0.5]
        self.log("pos_prediction_avg", torch.mean(pos_predictions), prog_bar=True)
        self.log("neg_prediction_avg", torch.mean(neg_predictions), prog_bar=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        xb, yb = val_batch
        pred = self(xb)
        loss = F.binary_cross_entropy(pred, yb.unsqueeze(1))
        self.log("val_loss", loss, batch_size=len(xb))

    def test_step(self, test_batch):
        xb, yb = test_batch
        y_pred = self(xb)
        self.test_probs.append(y_pred)
        self.test_targets.append(yb.unsqueeze(1).int())
        self.auroc_metric(y_pred, yb.unsqueeze(1).int())
        return {}

    def on_test_epoch_end(self):
        """
        Summarize testing statistics (AUROC and ROC) at end of testing and adds them to logging.
        The ROC curve is added as figure to self.logger.
        """
        probs = torch.cat(self.test_probs, dim=0)
        targets = torch.cat(self.test_targets, dim=0)

        auroc = self.auroc_metric.compute()
        self.log("test_auroc", auroc, on_step=False, on_epoch=True)

        self.roc_metric.update(probs, targets)
        fpr, tpr, thresholds = self.roc_metric.compute()

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

    def forward(self, input, optimized=False):
        """
        Takes an iterable of ete3.Tree as input and outputs a tensor of 0's and 1's to
        indicate whether each input tree is maximally parsimonious or not, respectively,
        for the sequences assigned to the leaf nodes.
        Args:
            input (list of Trees): has attribute to_parent["feature_0"] on each node,
                which is a torch tensor that encodes the mutation between the node and
                its parent, e.g. A -> G is encoded by [-1, 1, 0, 0]
            optimized (boolean): if True, runs more efficiently by skipping type check
                that allows ete.Tree input
        """
        if not optimized:
            if type(input) == Tree:
                logit = self.forward_on_tree(input)
                return F.sigmoid(logit)
        # assume input is a list (or iterable) of trees
        logits = torch.cat([self.forward_on_tree(item) for item in input])
        return F.sigmoid(logits)
        # return torch.stack(logits, dim=0)

    def forward_on_tree(self, tree: Tree):
        """
        Takes an ete3.Tree as input and outputs a 0 or 1 to indicate whether the input
        tree is maximally parsimonious or not, respectively, for the sequences assigned
        to the leaf nodes.
        Args:
            tree (ete3 Tree): each node has a torch tensor attribute
                to_parent["feature_0"], of size (n_sites, 4), that encodes the mutation
                between the node and its parent, e.g. A -> G is encoded by
                [..., [-1, 1, 0, 0], ...]
        """
        encoder_input = self.tree_traversal_mlp(tree, len(tree.sequence))
        encoder_output = self.site_aggregation(encoder_input)
        logit = self.classifier(encoder_output[0])
        return logit

    def tree_traversal_mlp(
        self,
        tree: Tree,
        seq_length,
        feature_name="feature_0",
    ):
        """
        Takes an ete3.Tree as input and outputs an encoding of the root sequence
        Args:
            tree (ete3 Tree): each node has a torch tensor attribute
                to_parent["feature_0"] that encodes the mutation between the node and
                its parent, e.g. A -> G is encoded by [-1, 1, 0, 0]
            seq_length (int): length of input sequences
            feature_name (string): name of feature assigned to nodes of the tree that
                are used for encoding
        """
        # root-ward traversal
        for node in tree.traverse(strategy="postorder"):
            if node.is_leaf():
                node.to_parent["feature_1"] = torch.zeros((seq_length, 4))
            elif len(node.children) == 1:  # node is root with single child
                assert node.up is None
                child = node.children[0]
                node.to_parent["feature_1"] = child.to_parent["feature_1"]
            else:
                feature_1 = torch.zeros((seq_length, 4))
                for i in range(seq_length):
                    try:
                        child1, child2 = node.children
                    except ValueError:
                        raise ValueError(
                            f"Input tree must be bifurcating, but node has"
                            "{len(node.children)} children"
                        )
                    left_feature_0 = child1.to_parent[feature_name][i]
                    left_feature_1 = child1.to_parent["feature_1"][i]
                    right_feature_0 = child2.to_parent[feature_name][i]
                    right_feature_1 = child2.to_parent["feature_1"][i]
                    left_data = torch.cat((left_feature_0, left_feature_1), dim=0)
                    right_data = torch.cat((right_feature_0, right_feature_1), dim=0)
                    feature_1[i] = self.node_aggregate(left_data, right_data)
                node.to_parent["feature_1"] = feature_1
        return tree.to_parent["feature_1"]

    def site_aggregation(self, input_features):
        """
        Takes an encoding of the root sequence of a tree and aggregates its n_sites
        using a Transformer
        """
        encoder_input = input_features.unsqueeze(1)  # batch_size = 1
        out = self.encoder(encoder_input)
        return out

    def node_aggregate(self, left_data, right_data):
        """
        takes in concatenation of feature vectors from two children of a given node, and
        outputs the `feature_1` vector for that node
        previous version: pass concatentation of feature vectors in both orders,
            `(left, right)` and `(right, left)` and add outputs, to apply symmetry
            constraint
        """
        output = self.up_traverse_stack(torch.cat((left_data, right_data)))
        # output += self.up_traverse_stack(torch.cat((right_data, left_data)))
        return output.unsqueeze(dim=0)


class TransformerEncoderTraversal(TraverseNN):
    """
    A pytorch module which takes a list of ete3.Trees as input and outputs 0's and 1's
    to indicate whether each input tree is maximally parsimonious or not, respectively,
    for the sequences assigned to the leaf nodes.

    The forward function first encodes the mutation features using a transformer encoder
    and then applies two traversals to the input tree, first root-ward and then leaf-ward

    For now, we only implement the root-ward traversal.

    Attributes:
        up_traverse_stack
        final
    """

    def forward_on_tree(self, tree: Tree):
        """
        Takes an ete3.Tree as input and outputs a 0 or 1 to indicate whether the input
        tree is maximally parsimonious or not, respectively, for the sequences assigned
        to the leaf nodes.
        Args:
            tree (ete3 Tree): each node has a torch tensor attribute
                to_parent["feature_0"] that encodes the mutation between the node and
                its parent, e.g. A -> G is encoded by [-1, 1, 0, 0]
        """
        tree = self.site_aggregation(tree)
        output = self.tree_traversal_mlp(tree, 1, feature_name="encoding")
        logit = self.classifier(output)
        return logit

    def site_aggregation(self, tree: Tree):
        # transform input to tensor of correct format for TransformerEncoder
        input = [
            node.to_parent["feature_0"] for node in tree.traverse(strategy="postorder")
        ]
        input = torch.stack(input)
        input = input.transpose(
            0, 1
        )  # swap first two dimensions -> [seq_length, batch_size, d_model]
        # we have one batch containing sequences for all nodes
        out = self.encoder(input)  # TransformerEncoder
        # assign learned features to
        for node in tree.traverse(strategy="postorder"):
            node.to_parent["encoding"] = out[0][1].unsqueeze(0)
        return tree
