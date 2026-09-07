# Genesis Playground

Simulation environment for DREAM project. 

## Dependencies
- Ubuntu 22.04 (Tested) 
- Python 3.10.12 (Tested)

## Installation
This installtion guide will clone this repository and initialize a genesis-world submodule. \
Dependencies (pyyaml, pyvista, torch, torchvision) will also be installed in the virtual environment. \
NOTE: genensis-world submodule is fixed to the commit of release tag v1.4.0 (might change with future updates)
```
# Clone the repository
git clone --recurse-submodules https://github.com/BjarkeHJ/genesis_playground.git genesis_playground

# Enter repository
cd genesis_playground

# Creates virtual environment (.venv) and builds dependencies
source build_deps.sh
```

## Usage
This example runs a basic environment with a drone and a payload connected by the TetherModel. \
The drone will attempt to fly but crash, since propeller RPM is fixed - no control. 
```
# Enter repository
cd genesis_playground

# Activate virtual environment
source activate_venv.sh

# Run example
python workspace/examples/cable_testing/test_sim.py
```

## Force reinstallation
To force a reinstallation, simply delete the virtual environment and re-build it. 
```
# Enter repository
cd genesis_playground

# Delete virtual environment (Everything is installed in here - Python only)
rm -fr .venv

# Reinstall in a new virtual environment
source build_deps.sh
```