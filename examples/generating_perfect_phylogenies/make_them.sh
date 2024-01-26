#!/bin/bash

# The max-sites parameter, in each example, is set to the minimum value which returns at 
# least one phylogeny (labelled tree)

# To visualize these trees, load one up as an ete3 Tree and run something like:
# print(tree.get_ascii(attributes=["sequence", "subs"])). This will print a nice
# ascii tree where each node is labelled with its sequence and substititions.

echo "Requiring distinct leaf sequences"
../../scripts/make_phylogenies.py example1.nwk ex1_output_1_site_unique_leaves.nwk \
    --unique-leaves --max-sites 1
# Yields 216 phylogenies.

echo "Requiring subs on all non-terminal edges + unique leaf sequences"
../../scripts/make_phylogenies.py example1.nwk ex1_output_1_site_subs_on_internal.nwk \
    --unique-leaves --sub-on-all-internal --max-sites 1
# Yields 120 phylogenies.

echo "Requiring subs on all edges..."
../../scripts/make_phylogenies.py example1.nwk ex1_output_2_sites_subs_on_edges.nwk \
    --sub-on-all-edges --max-sites 2 
# Yields 13,248 phylogenies.

echo "Requiring distinct leaf sequences"
../../scripts/make_phylogenies.py example2.nwk ex2_output_2_sites_unique_leaves.nwk \
    --unique-leaves --max-sites 2 
# Yields 1,268,352 phylogenies.

echo "Requiring subs on all non-terminal edges + unique leaf sequences"
../../scripts/make_phylogenies.py example2.nwk ex2_output_2_site_subs_on_internal.nwk \
    --unique-leaves --sub-on-all-internal --max-sites 2
# Yields 179,136 phylogenies.

echo "Requiring subs on all edges..."
../../scripts/make_phylogenies.py example2.nwk ex2_output_3_sites_subs_on_edges.nwk \
    --sub-on-all-edges --max-sites 3
# Yields 15,482,880 phylogenies.

