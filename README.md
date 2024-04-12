# dpvt
Deep (neural networks for) Phylogenetics Via Traversals

## Installation

```bash
mamba env create -f environment.yml
```


## Training Workflow

We have a workflow implemented in Snakemake (`Snakefile`), which takes as input in `config.yaml` names of models (see *Neural Network Model*) and datasets (see *Training Data*) and trains and evaluates the given models on all given datasets.

To execute the workflow `snakemake -c[num_cores]`, where `[num_cores]` should be replaced with the number of cores you want to use.


### Hyperparameter Optimization

By default, running the workflow with Snakemake will perform hyperparameter tuning with optuna for all models and datasets.
All models tested are saved in `hyper_checkpoints/`, which also contains `json` files with the best hyperparameters.
These are then used for model training.
If the files with best hyperparameters exist already for a given model and dataset, running the workflow will skip hyperparameter tuning.


## Training Data

### Generating Perfect Phylogenies (not yet tested for correctness or completeness)
A python class for creating perfect phylogenies given a topology is in 
`scripts/perfect_phylogeny.py`.

A command line interface is provided by `scripts/make_phylogenies.py`.

Example usage is in `examples/generating_perfect_phylogenies`

Be careful, there are many perfect phylogenies even for a very small topology.

### Perturbing the phylogenies
...coming soon...


### Data format

We currently assume that training/testing/validation data is pickled as one dictionary with keys being trees and values determining whether the tree is a MP tree (label `0`) or not (label `1`).
Our training/validation/testing data split is 0.6/0.2/0.2 and when splitting the data we ensure that we get balanced training, validation, and testing set.


## Neural network model


### TraverseNN

We define a Pytorch module `TraverseNN` which evaluates whether edges in a given labeled tree appear in a maximum parsimony tree, for the given sequences on the leaf nodes.
This module is defined in `dpvt/traverse_nn.py`.

The module first performs a tree traversal:

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

After the tree traversal we us a transformer encoder in `site_aggregation()` to aggregate the per-site information we learned from the tree traversal into a single value for classifying whether an edge is in a MP tree or not.

[At the current stage, we only make predictions to determine whether the tree is a MP tree or not]


### TransformerEncoderTraversal

This Pytorch module inherits from `TraverseNN` and changes the order of the steps described for this module to first aggregate per-site information at every node of a tree and then use the learned features for the tree traversal.


## Logging training

To train the model, run `snakemake --cores 1` in the directory `dpvt/train`.

To view training logs, run `tensorboard --logdir .` and direct your browser to `http://localhost:6006/`.
The tensorboard additionally shows ROC curves for the performance of classification on the test set.


## File structure of this repo

- `train`: contains `Snakefile` and `config.yaml`, in which models and datasets for training are specified.
- `neural_network`: contains `models.py`, in which models are defined, and `wrapper.py`, containing wrappers for these models.
- `dpvtex`: contains `dpvt_data.py`, which implements functions to get datasets for a given nickname and `dpvt_zoo.py`, which creates models for a given nickname. These nicknames are provided to the `Snakefile` in `config.yaml`.
- `generate_data.py`: contains files for generating training and validation data. The data should be saved in data. Data generation is independent of the workflow in `Snakefile`.
