# Code for the paper "Fourier Neural Networks as Function Approximators and Differential Equation Solvers"
Available at https://onlinelibrary.wiley.com/doi/abs/10.1002/sam.11531

1. The architecture of the paper is implemented in `neural_network_fourier.py`, if you want to compare with a neural network without the regularization and initialization schemes, comment line 58 in `codes/FourierCoeff.py` and uncomment line 59.
2. Increase N_points in `codes/dataset_prod.m` and set `batchsize=N_points` in `codes/FourierCoeff.py` then do `python3 FourierCoeff.py` to run.
3. To change (m = stdev), go to `codes/neural_networks_fourier.py` and change `stdev at line 22` (w01) (m = stdev)


# Citation

```
@article{Ngom2021,
author = {Ngom, M. and Marin, O.},
title = {Fourier neural networks as function approximators and differential equation solvers},
journal = {Statistical Analysis and Data Mining: The ASA Data Science Journal},
volume = {14},
number = {6},
pages = {647-661},
keywords = {differential equations, Fourier decomposition, neural networks},
doi = {https://doi.org/10.1002/sam.11531},
url = {https://onlinelibrary.wiley.com/doi/abs/10.1002/sam.11531},
eprint = {https://onlinelibrary.wiley.com/doi/pdf/10.1002/sam.11531},
abstract = {Abstract We present a Fourier neural network (FNN) that can be mapped directly to the Fourier decomposition. The choice of activation and loss function yields results that replicate a Fourier series expansion closely while preserving a straightforward architecture with a single hidden layer. The simplicity of this network architecture facilitates the integration with any other higher-complexity networks, at a data pre- or postprocessing stage. We validate this FNN on naturally periodic smooth functions and on piecewise continuous periodic functions. We showcase the use of this FNN for modeling or solving partial differential equations with periodic boundary conditions. The main advantages of the current approach are the validity of the solution outside the training region, interpretability of the trained model, and simplicity of use.},
year = {2021}
} ```

