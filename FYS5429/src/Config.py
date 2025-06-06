import torch

PARAMS = {
    "hidden_dim": 64,          # Amount of hidden nodes in the neural network 
    "n_layers": 3,             # Amount of layers in the neural network
    "act_fn": torch.nn.GELU(), # Ensure torch is imported in this module if necessary.
    "learning_rate": 1e-4,     # Learning rate to the neural network
    "N_collocation": 2000,     # Number of samples per epoch during training
    "n_epochs":3000,           # Number of epochs during training
    "n_epochs_norm": 200,      # Number of epochs normalizing the neural network as pretraining
    "E": 0.44079,              # Energy estimate from DMC
    "V": 1,                    # Interaction strength
    "d": 2,                    # Dimension of system
    "device": "cpu",           # Processing unit
    "n_particles": 2,          # Number of particles
    "nx": 1,                   # Number of Basis Functions
    "ny": 1,                   # ny = nx
    "dimensions": 2,           # Dimensionality of the system
    "omega": 0.1,              # Harmonic Oscillator trap frequency
    "L": 8.0,                  # Length of Grid calculating the initial Hartree Fock Slater determinant
    "L_E": 9.0,                # Length of Plotting/Calculating the Energy
    "n_grid": 30,              # Amount of points calculating 
    "batch_size": int(1e3),    # Batching the energy calculation
    "n_samples": int(1e5)      # Amount of points calculating the Energy
}
