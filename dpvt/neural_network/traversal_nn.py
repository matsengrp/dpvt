import torch
from torch import nn
from ete3 import Tree


class TraverseNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.up_traverse_stack = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32,4),
        )
        # self.down_traverse_stack = nn.Sequential(
        #     nn.Linear(8, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 4),
        # )
        self.final = nn.Linear(4, 1)

        # self.loss = nn.BCEWithLogitsLoss()

    def forward(self, tree):
        """
        Args:
            tree (ete3 Tree): has attribute feature_0 on each node, which is a torch
                tensor that encodes the mutation between the node and its parent, e.g. 
                A -> G is encoded by [-1, 1, 0, 0]
        """
        tree = tree.copy()
        # root-ward traversal
        for node in tree.traverse(strategy="postorder"):
            if node.is_leaf(): 
                node.feature_1 = torch.zeros(4)
            else:
                try:
                    child1, child2 = node.children
                except ValueError:
                    raise ValueError("Input tree must be bifurcating")
                x = torch.cat((child1.feature_0, child1.feature_1))
                node.feature_1 = self.up_traverse_stack(x)
                y = torch.cat((child2.feature_0, child2.feature_1))
                node.feature_1 += self.up_traverse_stack(y)
        # leaf-ward traversal -> skip for now
        # logits = self.up_traverse_stack(x)
        # feed root feature into final layer
        logit = self.final(tree.feature_1)
        return logit