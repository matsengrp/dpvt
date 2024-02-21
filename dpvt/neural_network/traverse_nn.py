import torch
from torch import nn
import torch.nn.functional as F
import lightning as L
from ete3 import Tree


class TraverseNN(L.LightningModule):
    """
    A pytorch module which takes a list of ete3.Trees as input and outputs 0's and 1's
    to indicate whether each input tree is maximally parsimonious or not, respectively,
    for the sequences assigned to the leaf nodes.

    The forward function applies two traversals to the input tree, first root-ward and
    then leaf-ward.

    For now, we only implement the root-ward traverseal.

    Attributes:
        up_traverse_stack
        final
    """

    def __init__(self):
        super().__init__()
        # learning rate
        self.lr = 0.05
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
        self.final = nn.Linear(4, 1)

        # self.loss = nn.BCEWithLogitsLoss()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        xb, yb = train_batch
        pred = torch.cat([self.forward_on_tree(item) for item in xb])
        loss = F.binary_cross_entropy_with_logits(pred, yb)
        self.log("train_loss", loss, batch_size=len(xb), on_epoch=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        xb, yb = val_batch
        pred = torch.cat([self.forward_on_tree(item) for item in xb])
        loss = F.binary_cross_entropy_with_logits(pred, yb)
        self.log("val_loss", loss, batch_size=len(xb))

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
                return logit
        # assume input is a list (or iterable) of trees
        logits = torch.cat([self.forward_on_tree(item) for item in input])
        return logits
        # return torch.stack(logits, dim=0)

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
        # tree = tree.copy()
        # root-ward traversal
        for node in tree.traverse(strategy="postorder"):
            if node.is_leaf():
                node.to_parent["feature_1"] = torch.zeros(4)
            elif len(node.children) == 1:  # node is root with signle child
                assert node.up is None
                child = node.children[0]
                node.to_parent["feature_1"] = child.to_parent["feature_1"]
            else:
                try:
                    child1, child2 = node.children
                except ValueError:
                    raise ValueError("Input tree must be bifurcating")
                left_feature_0 = child1.to_parent["feature_0"]
                left_feature_1 = child1.to_parent["feature_1"]
                right_feature_0 = child2.to_parent["feature_0"]
                right_feature_1 = child2.to_parent["feature_1"]
                left_data = torch.cat((left_feature_0, left_feature_1))
                right_data = torch.cat((right_feature_0, right_feature_1))
                node.to_parent["feature_1"] = self.node_aggregate(left_data, right_data)
        # leaf-ward traversal -> skip for now
        # logits = self.up_traverse_stack(x)
        # feed root feature into final layer
        logit = self.final(tree.to_parent["feature_1"])
        return logit

    def node_aggregate(self, left_data, right_data):
        """
        pass concatentation of feature vectors in both orders, `(left, right)` and
        `(right, left)` and add outputs, to apply symmetry constraint
        """
        output = self.up_traverse_stack(torch.cat((left_data, right_data)))
        output += self.up_traverse_stack(torch.cat((right_data, left_data)))
        return output
