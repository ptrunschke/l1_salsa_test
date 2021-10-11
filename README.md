# Convergence bounds for nonlinear least squares and applications to tensor recovery
![](https://img.shields.io/badge/linux--64-gray?label=platform&labelColor=gray&color=lightgray&style=flat)
![](https://img.shields.io/github/license/ptrunschke/l1_salsa_test)
[![arXiv](https://img.shields.io/badge/arXiv-2108.05237-b31b1b.svg)](https://arxiv.org/abs/2108.05237)


This repository contains the code for the numerical experiments performed in the paper [**Convergence bounds for nonlinear least squares and applications to tensor recovery**](https://arxiv.org/abs/2108.05237).
If you find this repository helpful for your work,  please kindly cite the paper.
<pre>
@misc{trunschke2021convergence,
      title={Convergence bounds for nonlinear least squares and applications to tensor recovery},
      author={Philipp Trunschke},
      year={2021},
      eprint={2108.05237},
      archivePrefix={arXiv},
      primaryClass={math.NA}
}
</pre>


## Sample generation
The samples that were used for the numerical experiment in the paper are provided in this repository.

### Usage
To generate new samples, first create a directory (in the following referred to as `PROBLEM_DIRECTORY`) that contains the problem description file `parameters.json`.
Then execute the following commands to draw `10000` samples of the quantity of interst.
```
python compute_samples.py PROBLEM_DIRECTORY 10000
python compute_functional.py PROBLEM_DIRECTORY
```
For more details see `vmc_sampling/README.md`.

### Dependencies
- numpy
- scipy
- joblib
- fenics


## Best approximation

### Usage
Execute one of the following scripts to replicate the corresponding numerical experiment.

|       | Script |
|-------|--------|
|Table 1|`l1_salsa_test_darcy.py`|
|Table 2|`l1_salsa_test_darcy_lognormal.py`|
|Table 3|`l1_salsa_test_darcy_lognormal10.py`|

### Dependencies
- numpy
- scipy
- [xerus](https://libxerus.org/) (branch: `SALSA`)
- scikit-learn
- matplotlib
- colored
- rich


## Installing all dependencies
You can run the accompanying bash script to install all dependencies in the new `conda` environment `ENVNAME`.
```
bash install.sh ENVNAME
```
