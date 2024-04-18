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

To execute the workflow, run `snakemake -c[num_cores]` in the directory `dpvt/train`, where `[num_cores]` should be replaced with the number of cores you want to use.
Alternatively, run `snakemake --snakefile dpvt/train/Snakefile -c[num_cores]` in the root directory, or from any directory with the `--snakefile` path argument replaced as appropriate.


## Training Data

### Generating Perfect Phylogenies
A python class for creating perfect phylogenies given a tree topology is in 
`dpvt/generate_data/perfect_phylogeny.py`. 

Call the `make_phylogenies` method to generate all perfect phylogenies (with a certain 
minimality condition) for a topology. Call the `make_random_phylogeny` method for a single 
perfect phylogeny (the distribution currently is not uniform, see 
`dpvt/tests/test_random_phylogenies.py`).

Example usage for generating all perfect phylogenies for a topology is in 
`examples/generating_perfect_phylogenies/make_them.sh` Be careful, there are many 
perfect phylogenies even for a very small topology.

Run time is still sub-optimal. However, generating random perfect phylogenies with 25
or so leaves is doable.

### Perturbing the phylogenies
Perturbing trees is handled by `dpvt/generate_data/perturb_phylogeny.py`. See
`examples/generating_perfect_phylogenies/perturb_random_perfect_phylogenies.py` for an
example of generating random perfect phylogenies and perturbing them to obtain a similar 
phylogeny, but with worse parsimony score.

## Neural network model

We define a Pytorch module `TraverseNN` which evaluates whether edges in a given labeled tree appear in a maximum parsimony tree, for the given sequences on the leaf nodes.
This module is defined in `dpvt/neural_network/models.py`.

The model works as follows:

0. We assume an input tree has a `sequence` attribute on each node, which is a string consisting of the characters `A`, `G`, `C`, `T`.

1. Edge mutation annotation: At each node, we assign a `edge_mutation` attribute which encodes the difference between the node's `sequence` and its parent's `sequence`. The `edge_mutation` attribute is a pytorch tensor of dimension `(n_sites, 4)`. 
A mutation `A -> T` from parent to child is encoded as `[..., [-1, 1, 0, 0], ...]`.

2. Traversal step: We apply two traversals to the tree, combining mutation data across the tree. This step applies to each site separately.  
    - Post-order traversal: We first traverse the tree root-ward, where at each step we assign a node the attribute `node.to_parent["clade_mutation"]`, a tensor of dimension `(n_sites, 4)`,  which is the output of a single-hidden-layer neural network, stored in the class attribute `traverse_stack`.
    As input, the neural network takes the `edge_mutation` and `clade_mutation` features of its two children nodes, and applies symmetrization so that the order of the children does not matter.
    For leaf nodes, which have no children, `node.to_parent["clade_mutation"]` is initialized to a zero tensor.  

    - Pre-order traversal: We traverse the tree leaf-ward, where at each step we assign a node the attribute `node.from_parent["clade_mutation"]`, a tensor of dimension `(n_sites, 4)`, which is the output of the single-hidden-layer neural network `traverse_stack`. 
    As input, the neural network takes the `node.from_parent[edge_mutation]` and `node.from_parent[clade_mutation]` tensors of the parent node and the `node.to_parent[edge_mutation]` and `node.to_parent[clade_mutation]` tensors of the sister node.

3. Site-aggregation step: We apply a transformer encoder to combine the clade mutation data across sites.
This uses the `encoder` attribute.
As input, we concatenate the tensors `node.to_parent["clade_mutation"]` and `node.from_parent["clade_mutation"]`, to form a tensor of dimension `(n_sites, 8)`. 
This is passed through the transformer encoder, and the first row, a size-`8` tensor, is kept.
These tensors from each node are stacked together in preorder-traversal order, forming a tensor of dimension `(n_nodes, 8)`.

4. Final output step: The output from the previous step is passed through a linear layer, the `classifier` attribute, to produce a tensor in logit space, of dimension `(n_nodes)`. 
The a sigmoid is applied.
At entry $i$, values near `0.0` mean the $i$-th edge is in a maximum parsimony tree, while values near `1.0` mean the $i$-th edge is not in a maximum parsimony tree. 
The output values are arranged to correspond to edges in preorder traversal order.



## Logging training

To view training logs, run `tensorboard --logdir .` and direct your browser to `http://localhost:6006/`.


## File structure of this repo

- `train`: contains `Snakefile` and `config.yaml`, in which models and datasets for training are specified.
- `neural_network`: contains `models.py`, in which models are defined, and `wrapper.py`, containing wrappers for these models.
- `dpvtex`: contains `dpvt_data.py`, which implements functions to get datasets for a given nickname and `dpvt_zoo.py`, which creates models for a given nickname. These nicknames are provided to the `Snakefile` in `config.yaml`.
- `generate_data.py` contains files for generating training and validation data. The data should be saved in data. Data generation is independent of the workflow in `Snakefile`.
