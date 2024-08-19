# Description
This repository houses the code used to create the plots used in "A unified framework to enforce, discover, and promote symmetry in machine learning" by Otto et al. A preprint can be found on the arXiv (https://arxiv.org/abs/2311.00212). 

# Getting started
To run this code, download the repository, and run the following to install `symm_ml` as a python package in your local environment.

```
pip install -r requirements.txt
pip install -e .
```

The code used to generate the data for the numerics examples can be found in `symm_ml.numerics.experiment`. Simply change the experiment variables at the top of the file and run `python experiment.py`. NOTE: The code was not optimized to be memory efficient and these optimization problems can quickly grow. Be mindful of this when setting the number of workers and size of the problems.