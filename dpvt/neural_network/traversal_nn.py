from torch import nn
from ete3 import Tree


class TraverseNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32,4),
        )
        self.down_traverse_stack = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits