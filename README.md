# dpvt
Deep (neural networks for) Phylogenetics Via Traversals

## Installation

```bash
mamba env create -f environment.yml
```

To install this package locally, clone the repo and in the root folder (with the `setup.py` file) run:
```bash
pip install -e .
```


## Training Workflow

We have a workflow implemented in Snakemake (`Snakefile`), which takes as input in `config.yaml` names of models (see *Neural Network Model*) and datasets (see *Training Data*) and trains and evaluates the given models on all given datasets.

To execute the workflow, run `snakemake -c[num_cores]` in the directory `dvpt/train`, where `[num_cores]` should be replaced with the number of cores you want to use.


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

We define a Pytorch module `TraverseNN` which evaluates whether edges in a given labeled tree appear in a maximum parsimony tree, for the given sequences on the leaf nodes.
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




## Logging training

To view training logs, run `tensorboard --logdir .` and direct your browser to `http://localhost:6006/`.


## File structure of this repo

- `train`: contains `Snakefile` and `config.yaml`, in which models and datasets for training are specified.
- `neural_network`: contains `models.py`, in which models are defined, and `wrapper.py`, containing wrappers for these models.
- `dpvtex`: contains `dpvt_data.py`, which implements functions to get datasets for a given nickname and `dpvt_zoo.py`, which creates models for a given nickname. These nicknames are provided to the `Snakefile` in `config.yaml`.
- `generate_data.py` contains files for generating training and validation data. The data should be saved in data. Data generation is independent of the workflow in `Snakefile`.
