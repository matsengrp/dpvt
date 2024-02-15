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