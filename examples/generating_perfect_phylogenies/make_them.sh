#!/bin/bash

# The maximum number of sites is the minimum with results.

echo "Requiring distinct leaf sequences"
../../scripts/make_phylogenies.py example1.nwk ex1_output_1_site_unique_leaves.nwk \
    --unique-leaves --max-sites 1
# Yields 216 topologies.

echo "Requiring subs on all non-terminal nodes + unique leaf sequences"
../../scripts/make_phylogenies.py example1.nwk ex1_output_1_site_subs_on_internal.nwk \
    --unique-leaves --sub-on-all-internal --max-sites 1
# Yields 120 topologies.


echo "Requiring subs on all edges..."
../../scripts/make_phylogenies.py example1.nwk ex1_output_2_sites_subs_on_edges.nwk \
    --sub-on-all-edges --max-sites 2 
# Yields 13,248 topologies.

echo "Requiring distinct leaf sequences"
../../scripts/make_phylogenies.py example2.nwk ex2_output_2_sites_unique_leaves.nwk \
    --unique-leaves --max-sites 2 
# Yields 1,268,352 topologies.

echo "Requiring subs on all edges..."
../../scripts/make_phylogenies.py example2.nwk ex2_output_3_sites_subs_on_edges.nwk \
    --sub-on-all_edges --max-sites 3
# Yields 15,482,880 topologies.


# To visualize these trees, load one up as an ete3 Tree and run something like:
# print(tree.get_ascii(attributes=["sequence", "sub"])). This will print a nice
# ascii tree where each node is labelled with its sequence and substititions.