#!/bin/bash


../../scripts/make_phylogenies.py example.nwk output_1_site.nwk --max-sites 1
../../scripts/make_phylogenies.py example.nwk output_1_sites_leaves.nwk --unique-leaves --max-sites 1
../../scripts/make_phylogenies.py example.nwk output_1_sites_leaves_cols.nwk --unique-leaves --max-sites 1
../../scripts/make_phylogenies.py example.nwk output_2_sites_leaves_cols.nwk --unique-leaves --max-sites 2
