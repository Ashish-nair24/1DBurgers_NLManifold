# 1DBurgers_NLManifold

This is a finite volume solver for the 1D Parameterised Burger's equation with an option to solve for reduced-order, approximate solutions using the Galerkin and LSPG projection methods. A non-linear manifold learnt using autoencoders(non-linear map) is used to map to and from the latent space.

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

The reduced-order solver uses Galerkin or LSPG projections(based on user choice) to evolve the latent variable (reduced set of equations). The user can choose between the Decoder Jacobian formulation using `Calculate ROM : True` or the Encoder Jacobian formulation (novel method) using `Encoder Jacobian Approximation : True`, in `paramsDictROM`. The former uses the Moore-Penrose psuedo-inverse of the jacobian of the decoder netwrok at each time instant to map the full-order RHS to the latent space. The former uses the jacobian of the encoder network for the same.

### Discrete Empirical Interpolation (DIEM)

The user may choose to hyper-reduce the ROM calculations by using QDIEM, which can be activated by setting `QDIEM Interpolation : True` in `paramsDictFOM`. The QDIEM formulation uses the pivoted QR decomposition of the snapshot array specified under `FOM Path`, to determine the sampling locations and also to learn the POD basis. Make sure that the snapshot array specified under `FOM Path` is either the snapshots of the full-order solution or the full-order RHS.

### Output and post-processing 

The outputs of the full-order and reduce-order runs are stored in the directory whose name is specified in `Output Directory :` under `paramsDictFOM`. Some post-processing options available are : 

1. `saveFOM` - Saves the full-order solution snapshots as a 2-Dimensional array in the `.npy` format.
2. `scaleFOM` - Centeres the full-order snapshots about the initial condition, scales it between [0,1] and re-orders it to the 'HCW' format for compatability with encoder/decoder training.
3. `saveROM` - Saves the approximate full-order solutions obtained from the reduced-order calculations in the `.npy` format.


### Utilities

The utilities included in **`utils/`** are :

1. **`offlineRecon.npy`** - Performs offline reconstruction of the full-order snapshots using the encoder/decoder netwroks stored in the **`Models/`** directory as a sanity check for the trained networks.
2. **`vis.npy`** - Calculates the %RAE error and generates visulisations based on the user input.


