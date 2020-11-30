# 1DBurgers_NLManifold

This is a finite volume solver for the 1D Paremetrised Burger's equation with an option to solve for reduced-order approximate solutions using the Galerkin and LSPG methods. A non-linear manifold learnt using autoencoders(non-linear map) is used to formulate the reduced set of equations.

# Dependencies

The code requires `numpy` to run the  full-order solver and `scipy` and `tensoflow` to run the reduced-order solver.

# Running the code

The full-order solver can be run by first specifying the solver parameters under `paramsDictFOM` within **`driver.py`** and running it.

The reduced-order solver can be run by first specifying the ROM solver parameters under `paramsDictROM` within **`driver.py`** and running it. Make sure that the encoder and decoder networks are stored the `.h5` format inside the **`Models/`** directory. The reduced-order solver requires a trained encoder and decoder, profiles for centering and normalization saved as `.npy` files.

## Table of Contents 
* Full-Order formulation
* Reduced-Order formulation
* Discrete Empirical Interpolation (DIEM)
* Output and post-processing
* Utilities

### Full-Order formulation

The full-order solver uses the Godunov's scheme to solve for the spatial fluxes at each time instant and the user can choose between a first-order accurate implicit or explicit scheme for time-integration. The parameter space can be defined by specifying `Parameters` in `paramsDictFOM`.

### Reduced-Order formulation 

The reduced-order solver uses Galerkin or LSPG projections(based on user choice) to evolve the latent variable (reduced set of equations). The user can choose between the Decoder Jacobian formulation or the Encoder Jacobian formulation (novel method). The former uses the Moore-Penrose psuedo-inverse of the jacobian of the decoder netwrok at each time instant to map the full-order RHS to the latent space. The former uses the jacobian of the encoder network for the same.
