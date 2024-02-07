#!/bin/bash

# The min-sites and max-sites parameters, in each example, are set to the minimum value 
# which returns at least one phylogeny (labelled tree).

# To visualize these trees, load one up as an ete3 Tree and run something like:
# print(tree.get_ascii(attributes=["sequence", "subs"])). This will print a nice
# ascii tree where each node is labelled with its sequence and substititions.


the_path=../../dpvt/scripts/make_phylogenies.py

echo "Requiring subs on all non-terminal edges..."
$the_path example1.nwk ex1_output_1_site_subs_on_internal.nwk --sub-on-all-internal \
    --max-sites 1
# Yields 156 phylogenies.

echo "Requiring subs on all non-terminal edges + unique leaf sequences..."
$the_path example1.nwk ex1_output_1_site_subs_on_internal_leaves.nwk --unique-leaves \
    --sub-on-all-internal --max-sites 1
# Yields 120 phylogenies.

echo "Requiring subs on all edges..."
$the_path example1.nwk ex1_output_2_sites_subs_on_edges.nwk --sub-on-all-edges \
    --min-sites 2 --max-sites 2 
# Yields 9504 phylogenies.

echo "Requiring subs on all non-terminal edges..."
$the_path example2.nwk ex2_output_2_sites_unique_leaves.nwk --sub-on-all-internal \
    --max-sites 1 
# Yields 24 phylogenies.

echo "Requiring subs on all non-terminal edges + unique leaf sequences..."
$the_path example2.nwk ex2_output_2_site_subs_on_internal.nwk --unique-leaves \
    --sub-on-all-internal --min-sites 2 --max-sites 2
# Yields 177,984 phylogenies.

echo "Requiring subs on all edges..."
$the_path example2.nwk ex2_output_3_sites_subs_on_edges.nwk --sub-on-all-edges \
    --min-sites 3 --max-sites 3
#Yields ??? phylogenies.

