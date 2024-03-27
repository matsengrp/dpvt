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

### Generating Perfect Phylogenies
A python class for creating perfect phylogenies given a topology is in 
`dpvt/scripts/perfect_phylogeny.py`. 

Call the `make_trees` method to generate all perfect phylogenies (with a certain 
minimality condition) for a topology. Call the `make_random_tree` method for a single 
perfect phylogeny (the distribution currently is not uniform, see 
`dpvt/tests/test_random_phylogenies.py`).

Example usage for generating all perfect phylogenies for a topology is in 
`examples/generating_perfect_phylogenies/make_them.sh` Be careful, there are many 
perfect phylogenies even for a very small topology.

Run time is still sub-optimal. However, generating random perfect phylogenies with 25
or so leaves is doable.

### Perturbing the phylogenies
Perturbing trees is handled by `dpvt/scripts/perturb_phylogeny.py`. See
`examples/generating_perfect_phylogenies/perturb_random_perfect_phylogenies.py` for an
example of generating random perfect phylogenies and perturbing them to obtain a similar 
phylogeny, but with worse parsimony score.

