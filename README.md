# dpvt
Deep (neural networks for) Phylogenetics Via Traversals

## Installation

```bash
mamba env create -f environment.yml
```
To install PyGeometric, which we may or may not use (I didn't put it in the environment 
file): 
```bash
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
```


## Training Data

### Generating Perfect Phylogenies (not yet tested for correctness or completeness)
A python class for creating perfect phylogenies given a topology is in 
`scripts/perfect_phylogeny.py`.

A command line interface is provided by `scripts/make_phylogenies.py`.

Example usage is in `examples/generating_perfect_phylogenies`

Be careful, there are many perfect phylogenies even for a very small topology.

### Perturbing the phylogenies
...coming soon...


## Neural network model

We define a Pytorch module `TraverseNN` which evaluates whether edges in a given labeled tree appear in a maximum parsiomy tree, for the given sequences on the leaf nodes.
This module is defined in `dpvt/traverse_nn.py`.

The module works as follows:

1. We assume the input tree has a `sequence` attribute on each node. 
For each node, the mutations between the node's sequence and its parent's sequence are encoded as a `torch.tensor` and stored as the attribute `node.to_parent["feature_0"]`
2. Post-order traversal step: We traverse the tree root-ward, where at each step we assign a node a 4-term tensor in the attribute `node.to_parent["feature_1"]`, which is the output of a single-hidden-layer neural network, stored in the module attribute `up_traverse_stack`.
As input, the neural network takes the `feature_0` and `feature_1` of its two children nodes, and applies symmetrization so that the order of the children does not matter.
For leaf nodes, which have no children, `node.to_parent["feature_1]` is initialized to a zero tensor. 
3. Pre-order traversal step: We traverse the tree leaf-ward, where at each step we assign a node a 4-term tensor in the attribute `node.from_parent["feature_1"]`, which is the output of the single-hidden-layer neural network `down_traverse_stack`. 
As input, the neural network takes the `feature_0` and `feature_1` tensors of the parent node and the sister node.
4. For each edge, we combine the tensors `node.to_parent["feature_1"]` and `node.from_parent["feature_1"]` using the linear layer in `TraverseNN.final`, to produce a logit. 
Negative values means the edge is in a maximum parsimony tree, while positive values means the edge is not in a maximum parsimony tree. 

[Current implementation does not do steps 3 and 4, for simplicity]


