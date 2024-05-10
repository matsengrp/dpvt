#!/bin/bash

# The min-sites and max-sites parameters, in each example, are set to the minimum value 
# which returns at least one phylogeny (labelled tree).

# To visualize these trees, load one up as an ete3 Tree and run something like:
# print(tree.get_ascii(attributes=["sequence", "subs"])). This will print a nice
# ascii tree where each node is labelled with its sequence and substititions.


SCRIPT=./make_all_perfect_phylogenies.py

echo "Requiring subs on all non-terminal edges..."
$SCRIPT example1.nwk ex1_output_1_site_subs_on_internal.nwk --max-sites 1
# Yields 60 phylogenies.

#echo "Requiring subs on all non-terminal edges + unique leaf sequences..."
#$SCRIPT example1.nwk ex1_output_1_site_subs_on_internal_leaves.nwk --unique-leaves \
#    --sub-on-all-internal --max-sites 1
# Yields 48 phylogenies.

echo "Requiring subs on all edges..."
$SCRIPT example1.nwk ex1_output_2_sites_subs_on_edges.nwk --sub-all-edges \
    --min-sites 2 --max-sites 2 
# Yields 1152 phylogenies.

echo "Requiring subs on all non-terminal edges..."
$SCRIPT example2.nwk ex2_output_2_site_subs_on_internal.nwk --max-sites 1 
#Yields 24 phylogenies.

#echo "Requiring subs on all non-terminal edges + unique leaf sequences..."
#$SCRIPT example2.nwk ex2_output_2_site_subs_on_internal_leaves.nwk --unique-leaves \
#    --sub-on-all-internal --min-sites 2 --max-sites 2
# Yields 134,784 phylogenies.

echo "Requiring subs on all edges..."
$SCRIPT example2.nwk ex2_output_3_sites_subs_on_edges.nwk --sub-all-edges \
    --min-sites 3 --max-sites 3
#Yields 8,736,768 phylogenies.
