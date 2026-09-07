#!/bin/bash

# Create and/or activate virtual environment
source activate_venv.sh 

# Install dependencies into virtual environment
pip install pyyaml || { echo "Error: failed to install pyyaml" >&2; $RET 1; }
pip install pyvista || { echo "Error: failed to install pyvista" >&2; $RET 1; }
pip install torch torchvision || { echo "Error: failed to install torch" >&2; $RET 1; }
cd external/genesis-world && pip install -e . || { echo "Error: failed to install genesis-world" >&2; $RET 1; }

cd -
