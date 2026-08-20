# Environment

Two conda environments are used, deliberately kept separate. They carry different versions of gemmi and numpy, so analysis scripts are not interchangeable between them — activate the right one for the step you are running.

Hardware: an NVIDIA GPU is required for BoltzGen and Boltz-2. Developed on an RTX A6000 (48 GB), driver CUDA 12.2. AutoDock Vina is CPU-only.

## boltz — scoring, docking and analysis

Used for Boltz-2 co-folding predictions, AutoDock Vina docking, and most analysis scripts.

| package | version |
|---|---|
| python | 3.12.13 |
| boltz | 2.2.1 |
| torch | 2.5.1+cu121 |
| pytorch-lightning | 2.5.0 |
| gemmi | 0.6.5 |
| numpy | 1.26.4 |
| pandas | 3.0.3 |
| vina | 1.2.7 |
| meeko | 0.7.1 |
| rdkit | 2026.3.3 |
| ProDy | 2.6.1 |
| matplotlib | 3.11.1 |

Setup:

    conda create -n boltz python=3.12
    conda activate boltz
    pip install boltz gemmi pandas numpy matplotlib
    pip install vina meeko prody

Note that `pip install vina` provides the **Python bindings only**, not a `vina` command-line binary. Docking is therefore driven from Python (`from vina import Vina`). The package is not available on conda-forge under that name.

## boltzgen — generative design

Used only for BoltzGen design runs.

| package | version |
|---|---|
| python | 3.12.13 |
| boltzgen | 0.3.2 (local checkout, not PyPI) |
| torch | 2.5.1+cu121 |
| pytorch-lightning | 2.6.5 |
| cuequivariance-torch | 0.10.0 |
| gemmi | 0.7.5 |
| numpy | 2.0.2 |
| pandas | 3.0.3 |

All BoltzGen runs in this project pass `--use_kernels false`, because the optimised kernels were incompatible with this CUDA/cuBLAS combination.

## System tools (not conda-managed)

| tool | path | used for |
|---|---|---|
| PyMOL | /usr/bin/pymol | building peptides from sequence (fab), rendering figures |
| OpenBabel | system | file conversion, PDB to PDBQT |

### PyMOL and conda conflict

PyMOL uses the system Python and **fails if a conda environment is active**, because conda takes precedence on PATH. The shell auto-activates base, so two deactivations are usually needed:

    conda deactivate
    conda deactivate
    which python3
    pymol -cq script.pml

Or bypass PATH for a single command:

    PATH=/usr/bin:/bin pymol -cq script.pml

Rendering is done headless with `-cq`, so no display or X11 forwarding is required.

## First-run downloads

Boltz-2 and BoltzGen fetch model weights and datasets from Hugging Face on first use. Boltz-2 also calls the ColabFold MSA server (https://api.colabfold.com) when `--use_msa_server` is set; this is an external service and does time out under load, which is not a fault of the local setup.
