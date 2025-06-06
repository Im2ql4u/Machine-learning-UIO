This repository contains Physics-Informed Neural Network (PINN) simulations for quantum systems,
organized clearly to separate core functionality, results, and configuration.

At the top level of the repository, you will find example notebooks and scripts placed directly
outside any folders for convenience and quick access. These are designed to be opened and executed
with JupyterLab.

Within the `src` folder:

- `Gausian/` contains all results and scripts for the 1D fermionic systems under a Gaussian potential.
- `QuantumDot/` contains results and scripts for quantum dot simulations.
- `master_functions/` contains all reusable code components:
    - `Slater_Determinant.py` defines all functions needed to use and analyze the slater determinant
    - `Physics.py` has functions embedding physical requirements, such as interactions
    - `Neural_Networks.py` has additional functions needed to evaluate, or train the PINN
    - `__init__.py` initializes the module structure.

- `PINN.py` defines the neural network architecture.
- `utils.py` provides supporting utility functions.
- `Config.py` defines all relevant parameters for the simulations.

Plotting configuration is handled by the `Thesis_style.mplstyle` file at the root. This custom style
file ensures consistent and high-quality plots throughout the project.

To run the code, simply open a terminal, navigate to the repository, and launch JupyterLab (if using JupyterLab) using:

    jupyter lab

Then open the desired notebook (e.g., `example_notebook.ipynb`) from the file explorer.

In notebooks, modules are loaded and configured dynamically like this:

```python
import os, sys
src_path = os.path.abspath("src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from Config import PARAMS
from master_functions import *
from PINN import PINN
import torch

# Apply parameters
PARAMS["n_particles"] = 2
PARAMS["omega"] = 0.5
PARAMS["nx"] = 1
PARAMS["E"] = 1.65977
PARAMS["n_epochs"] = 5000
PARAMS["N_collocation"] = 1000
PARAMS["L"] = 3
PARAMS["L_E"] = 4

torch.set_num_threads(1)
locals().update(PARAMS)
```

This setup allows full flexibility in parameter tuning and programmatic control directly from notebooks,
without requiring changes to the core code base. All simulations and results can be run and analyzed
through this interactive and reproducible interface.
