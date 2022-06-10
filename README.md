# false-vacuum-decay-desitter

Numerical implementation of the computations described in: arXiv:.....

## Gel'fand-Yaglom method

The Gel'fand-Yaglom method is implemented in Python together with Mathematica scripts that are called via bash. All 
intermediary results are written to CSV files.
(Python cannot supply the numerical precision needed for evaluating the associated Legendre functions involved.).
To get the results from the paper, consider the jupyter notebook `main.ipynb`.

The following will describe the structure of the implementation.
 - `BounceModel`: class for solving the differential equation for the bounce solution for a given set of initial
   conditions
 - `ModelShooter`: class for applying a shooting method (varying the initial position) to find the Coleman-deLuccia
   bounce
 - `VacuumFluctuation`: class to solve the analytical differential equation for the vacuum fluctuation (this class calls
   Wolframskript)
 - `EffectivePotential`: class to calculate the effective potential using the vacuum fluctuation (this class calls
   Wolframskript)
 - `GelfandYaglomModel`: class to solve the differential equation for the ratio-operator $T$ using the results from 
   the `VacuumFluctuation` and `EffectivePotential` classes.
 - `LargeLSolver`: class for calculating the ratio operator $T$ for a large number of $\ell$ values. This class also 
   deals with post-processing and plotting.

Depending on your system specs, there may be issues with calculating all vacuum fluctuation tables in one go. If the 
method `multiprocess_det_ratio` fails, try running it for separate `min_l` and `max_l` consecutively (e.g. from 2 to 
20, then from 20 to 40 etc.) with the option `recalculate_vacuum_fluctuation=True`. This will save all vacuum 
fluctuation tables to CSV files. Then, you can run the entire range with `recalculate_vacuum_fluctuation=True`.

The number of cores for the multiprocessing is set in `large_l_solver.py`.

## Green's function method

The Mathematica notebooks found in the `greens_method` folder contain an independent computation of the bounce,
functional determinant, WKB approximation for the homogeneous and divergent pieces of the one-loop contributions and the
bounce corrections as described in the coming paper.

The file `mainNotebook.nb` follows roughly the same sections as in the paper and uses the results from auxiliary
notebooks when needed, e.g. for WKB results or counterterms.

The computations concerning the functional determinant can be done in parallel using the Mathematica scripts in
the `cluster_scripts` folder. The data produced can be quite large and can be merged with the little python
script `rawDataMerger.py`, they must still be concatenated in different portions and loaded into the mainNotebook.nb for
analysis.
