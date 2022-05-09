# false-vacuum-decay-desitter

Numerical implementation of the computations described in: arXiv:.....

## Gel'fand-Yaglom method


## Green's function method

The Mathematica notebooks found in the `greens_method` folder contain an independent computation of the bounce, functional determinant, WKB approximation for the homogeneous and divergent pieces of the one-loop contributions and the bounce corrections as described in the coming paper. 

The file `mainNotebook.nb` follows roughly the same sections as in the paper and uses the results from auxiliary notebooks when needed, e.g. for WKB results or counterterms. 

The computations concerning the functional determinant can be done in parallel using the Mathematica scripts in the `cluster_scripts` folder. The data produced can be quite large and can be merged with the little python script `rawDataMerger.py`, they must still be concatenated in different portions and loaded into the mainNotebook.nb for analysis.
