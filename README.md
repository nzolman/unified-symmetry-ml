# Description
This repository houses the code used to create the plots used in "A unified framework to enforce, discover, and promote symmetry in machine learning" by Otto et al. A preprint can be found on the arXiv (https://arxiv.org/abs/2311.00212). 

# Getting started
To run this code, download the repository, and run the following to install `symm_ml` as a python package in your local environment.

```
pip install -r requirements.txt
pip install -e .
```

The code used to generate the data for the numerics examples can be found in `symm_ml.numerics.experiment`. Simply change the experiment variables at the top of the file and run `python experiment.py`. NOTE: The code was not optimized to be memory efficient and these optimization problems can quickly grow. Be mindful of this when setting the number of workers and size of the problems.

# Numerics Data
The numerics for calculating the recovery (or "statistical") dimension can be found in the form of `.csv` files under `data/`. Each file corresponds to the class of function and associated Lie algebra prior that was used ($T(n)$ for translations, or $SE(n)$ for the full special Euclidean group). A description of the columns can be found in the table below:

| Column name       | Description |
| --------          | ------- |
| $m$               | The input dimension   |
| $r$               | The number of latent features ($r \leq m$)|
| $d$               | The degree of the polynomial on the $r$ features|
| seed              | The random seed used for the experiment instantiation |
| n_recover         | The number of random samples needed to recover the function |
| n_lib             | The number of terms in the library (i.e. total number of coefficients)
| dt                | The amount of compute time needed to find the recovery dimension. |
| tot_iters         | The total number of `cvxpy` iterations needed to find the recovery dimension.