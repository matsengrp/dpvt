import torch
from torch import nn
from ete3 import Tree


class TraverseNN(nn.Module):
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
        self.up_traverse_stack = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32,4),
        )
        # self.down_traverse_stack = nn.Sequential(
        #     nn.Linear(16, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 4),
        # )
        self.final = nn.Linear(4, 1)

        # self.loss = nn.BCEWithLogitsLoss()

    def forward(self, input, optimized=False):
        """
        Takes an interable of ete3.Tree as input and outputs a tensor of 0's and 1's to 
        indicate whether each input tree is maximally parsimonious or not, respectively, 
        for the sequences assigned to the leaf nodes. 
        Args:
            input (Tree | list of Trees): has attribute feature_0 on each node, which is 
                a torch tensor that encodes the mutation between the node and its 
                parent, e.g. A -> G is encoded by [-1, 1, 0, 0]
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
            tree (ete3 Tree): has attribute feature_0 on each node, which is a torch
                tensor that encodes the mutation between the node and its parent, e.g. 
                A -> G is encoded by [-1, 1, 0, 0]
        """
        # tree = tree.copy()
        # root-ward traversal
        for node in tree.traverse(strategy="postorder"):
            if node.is_leaf(): 
                node.to_parent["feature_1"] = torch.zeros(4)
            else:
                try:
                    child1, child2 = node.children
                except ValueError:
                    raise ValueError("Input tree must be bifurcating")
                x_feature_0 = child1.to_parent["feature_0"]
                x_feature_1 = child1.to_parent["feature_1"]
                y_feature_0 = child2.to_parent["feature_0"]
                y_feature_1 = child2.to_parent["feature_1"]
                x = torch.cat((x_feature_0, x_feature_1))
                y = torch.cat((y_feature_0, y_feature_1))
                # pass concatentation of feature vectors of children in both orders,
                # `(x, y)` and `(y, x)` and add outputs, to apply symmetry constraint
                node.to_parent["feature_1"] = self.up_traverse_stack(
                    torch.cat((x, y))
                )
                node.to_parent["feature_1"] += self.up_traverse_stack(
                    torch.cat((y, x))
                )
        # leaf-ward traversal -> skip for now
        # logits = self.up_traverse_stack(x)
        # feed root feature into final layer
        logit = self.final(tree.to_parent["feature_1"])
        return logit

