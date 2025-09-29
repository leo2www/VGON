# VGON


VGON, short for “Variational Generative Optimization Network”, refers to a class of generative models intended to discover optimal solutions for quantum problems within the general framework of variational optimization.

# Project

Here we provide three numerical experiments corresponding to the quantum task discussed [in our paper](https://doi.org/10.1038/s42005-025-02261-4), as well as a unified Python environment configuration defined in  `pyproject.toml`.

For more configuration details, please  check the [setup documentation here](https://vgon.funqitang.cn/EnvSetup/) 

## Project Structure

The repository is organized as follows:

```
VGON/
├── BP/ # Barren Plateau experiments
│ ├── HXXZ/ # Heisenberg XXZ model
│ └── Z1Z2/ # Z1Z2 model
├── Degeneracy/ # Degeneracy detection experiments
│ ├── H232/ # H232 Hamiltonian
│ └── MG/ # Graph states
└── Gap/ # Nonlocality gap experiments
```

---

# Citing
If you are inspired by VGON for academic work, we encourage you to [cite our papers](https://doi.org/10.1038/s42005-025-02261-4). If you use VGON in industry, we'd love to hear from you over email.