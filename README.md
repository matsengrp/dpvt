# dpvt
Deep (neural networks for) Phylogenetics Via Traversals

## Installation

```bash
mamba env create -f environment.yml
```

To install this package locally, clone the repo and in the root folder (with the
`setup.py` file) run:
```bash
pip install -e .
```


## Training Data

Currently, datasets are stored in the `dpvt-experiments-1` repository.

### Data format

We currently assume that training/testing/validation data is pickled as one
dictionary with keys being trees and values being list of labels determining
whether an edge is in a MP tree (label `0`) or not (label `1`). These labels are
sorted according to a pre-order traversal. Our training/validation split is
0.8/0.2 and when splitting the data we ensure that we get balanced training and
validation sets, i.e. roughly the same ratio of MP to non-MP data in both sets.
We implement two different classes for datasets in `wrapper.py`:
`TraversalDataset` and `TreeDataset`. By default we use the `TraversalDataset`,
this is set in the `dpvt-experiments-1` repo.


#### TraversalDataset

- `traversal`: (node1, node2, node3) for all nodes node3 that are below an
	internal edge for every tree in the input. node1 and node2 will be input to
	the RNN predicting the feature of node3. For upward traversal this means that
	node1 and node2 are children of node3. Nodes are indexed by preorder
	traversal.
    - dimension: `(num_trees, 2, num_int_edges, 3)` (2 for upward and
	downward traversal)
- `mutations`: Contains for each tree, node, and site a tensor
	$(m_A,m_G,m_C,m_T)$, where $m_i=1$ and $m_j=-1$ if there is a mutation from
	base j to base i at this node, all other entries are $0$.
    - dimension:
	`(num_trees, num_nodes, num_sites, 4)`
- `labels`: For each tree and each node, indicates whether the edge above this
	node is in a MP tree (`0`) or not (`1`)
    - dimension: `(num_trees, num_nodes)`
- `masks`: For each tree and each node, indicates whether the edge above this
	node is an internal edge (`True`) or not (`False`)
    - dimension: `(num_trees,
	num_nodes)`

Note that if  input trees are of different sizes, `traversal`, `mutations`, and
`labels` are padded with `-1`, masks are padded with `False`.

When iterating through the TraversalDataset in the forward function, we stop as
soon as we see `-1` in the traversal tensor, as this means that we reached the
padding. And with the masks set to False for all those indiced, none of the `-1`
are being used for calculating the loss.


#### TreeDataset
- `data`: ete3 trees with node attributes `sequence`, which is needed to assess
  which mutations occur on each edge
  - dimension: `num_trees`
- `labels`: For each tree and each node, indicates whether the edge above this
  node is in a MP tree (`0`) or not (`1`)
  - dimension: `(num_trees, num_nodes)`
- `masks`: For each tree and each node, indicates whether the edge above this
  node is an internal edge (`True`) or not (`False`)
  - dimension: `(num_trees, num_nodes)`


## Neural network model


### TraverseNN

We define a Pytorch module `TraverseNN` which evaluates whether edges in a given
labeled tree appear in a maximum parsimony tree, for the given sequences on the
leaf nodes. This module is defined in `dpvt/neural_network/models.py`.

The model works as follows (for `TreeDataset`s):

0. We assume an input tree has a `sequence` attribute on each node, which is a
   string consisting of the characters `A`, `G`, `C`, `T`.

1. Edge mutation annotation: At each node, we assign a `edge_mutation` attribute
which encodes the difference between the node's `sequence` and its parent's
`sequence`. The `edge_mutation` attribute is a pytorch tensor of dimension
`(n_sites, 4)`. A mutation `A -> T` from parent to child is encoded as `[...,
[-1, 1, 0, 0], ...]`.

2. Traversal step: We apply two traversals to the tree, combining mutation data
   across the tree. This step applies to each site separately.  
    - Post-order traversal: We first traverse the tree root-ward, where at each
    step we assign a node the attribute `node.to_parent["clade_mutation"]`, a
    tensor of dimension `(n_sites, 4)`,  which is the output of a
    single-hidden-layer neural network, stored in the class attribute
    `traverse_stack`. As input, the neural network takes the `edge_mutation` and
    `clade_mutation` features of its two children nodes, and applies
    symmetrization so that the order of the children does not matter. For leaf
    nodes, which have no children, `node.to_parent["clade_mutation"]` is
    initialized to a zero tensor.  

    - Pre-order traversal: We traverse the tree leaf-ward, where at each step we
    assign a node the attribute `node.from_parent["clade_mutation"]`, a tensor
    of dimension `(n_sites, 4)`, which is the output of the single-hidden-layer
    neural network `traverse_stack`. As input, the neural network takes the
    `node.from_parent[edge_mutation]` and `node.from_parent[clade_mutation]`
    tensors of the parent node and the `node.to_parent[edge_mutation]` and
    `node.to_parent[clade_mutation]` tensors of the sister node.

3. Site-aggregation step: We apply a transformer encoder to combine the clade
mutation data across sites. This uses the `encoder` attribute. As input, we
concatenate the tensors `node.to_parent["clade_mutation"]` and
`node.from_parent["clade_mutation"]`, to form a tensor of dimension `(n_sites,
8)`. This is passed through the transformer encoder, and the first row, a
size-`8` tensor, is kept. These tensors from each node are stacked together in
preorder-traversal order, forming a tensor of dimension `(n_nodes, 8)`.

4. Final output step: The output from the previous step is passed through a
linear layer, the `classifier` attribute, to produce a tensor in logit space, of
dimension `(n_nodes)`. The a sigmoid is applied. At entry $i$, values near `0.0`
mean the $i$-th edge is in a maximum parsimony tree, while values near `1.0`
mean the $i$-th edge is not in a maximum parsimony tree. The output values are
arranged to correspond to edges in preorder traversal order.


If the `TraverseDataset` data structure is used, we save features in a tensor
and iterate through the `traversal` to compute the features. Everything else is
completely identical to how things are handled when using `TreeDataset`s.


### TransformerEncoderTraversal

This Pytorch module inherits from `TraverseNN` and changes the order of the
steps described for this module to first aggregate per-site information at every
node of a tree (step 3.) and then use the learned features for the tree
traversal (step 2.).

### TraverseMaxPooling

This model is very similar to `TraverseNN`, but we replace step 3. with a
simpler aggregation method. We aggregate sites by simply outputting as feature
the maximum of `feature[i]` over all sites.


### TraverseAvgPooling

This model is very similar to `TraverseNN`, but we replace step 3. with a
simpler aggregation method. We aggregate sites by simply outputting as feature
the average of `feature[i]` over all sites.


## Logging training

To view training logs, run `tensorboard --logdir .` and direct your browser to
`http://localhost:6006/`. The tensorboard additionally shows ROC curves for the
performance of classification on the test set.


## File structure of this repo

- `models.py`: contains definitions of models.

- `wrapper.py`: contains wrappers for a model and a dataset.

### File structure of companion repo `dpvt-experiments-1`

- `dpvtex`: contains `dpvt_data.py`, which implements functions to get datasets
  for a given nickname and `dpvt_zoo.py`, which creates models for a given
  nickname. These nicknames are provided to the `Snakefile` in `config.yaml`.
- `train`: contains `Snakefile` and `config.yaml`, in which models and datasets
  for training are specified.
