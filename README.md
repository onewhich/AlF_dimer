# Molecular dynamics-driven global tetra-atomic potential energy surfaces: Application to AlF-AlF complex

This repository contains the code and associated datasets for the machine learning of a potential energy surface for the purpose of molecular dynamics (MD) simulations, as implemented in [1]. For more detailed information, please refer to the manuscript [1].

### Main features

- Potential energy surface (PES) constructed for MD simulation, taking AlF-AlF complex as an example
- Regressor: Gaussian process regression (GPR)

#### Dependencies
scikit-learn==1.0.1

ase==3.22.1

pickle

### Usage
The codes were implemented using Python 3.8.

To train a machine learning potential energy surface (PES) model and make energy predictions, you can refer to the `run_AlF_dimer.py` script:
`$ python run_AlF_dimer.py`

The necessary datasets, including the training set and test set(s), are specified in `run_AlF_dimer.py`. For instance, `data/traj_train.xyz` and `data/traj_test.xyz` represent the training and test sets, respectively, in *extxyz (extended XYZ)* format. The reference energies in these datasets are expressed in electron volts (eV). The provided example training set, `data/traj_train.xyz`, consists of 18,732 AlF-AlF configurations combined from eight molecular dynamics (MD) trajectories. The test set,  `data/traj_test.xyz`, is derived from a single MD trajectory with 3,633 steps.

After training, the PES model is stored by default in `trained_ml_potential_model.pkl`. The training and testing results are printed in the  `AlF_dimer.log`file .

The structural representations of the AlF-AlF complex are computed in `representation.py`. These representations are then used as inputs to the Gaussian process regression (GPR) model for training. If you wish to modify the Gaussian process kernels, please refer to `machine_learning_potential.py`. Currently, the kernel is a combination of the Matern(5/2) kernel and a dot-product kernel, with a white noise kernel indicating the noise level of the training set.

[1] X. Liu, W. Wang, J. Pérez-Ríos, Molecular dynamics-driven global tetra-atomic potential energy surfaces: Application to AlF-AlF complex (2023)
