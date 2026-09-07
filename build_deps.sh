#!/bin/bash

if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    RET=return
else
    RET=exit
fi

if [ ! -d ".venv" ]; then
    echo "No .venv found, creating one..."
    python3 -m venv .venv || { echo "Error: failed to create .venv" >&2; $RET 1; }
fi

if [ ! -f ".venv/bin/activate" ]; then
    echo "Error: .venv/bin/activate not found, venv creation failed" >&2
    $RET 1
fi

source .venv/bin/activate

pip install pyyaml || { echo "Error: failed to install pyyaml" >&2; $RET 1; }
pip install pyvista || { echo "Error: failed to install pyvista" >&2; $RET 1; }
cd external/genesis-world && pip install -e . || { echo "Error: failed to install genesis-world" >&2; $RET 1; }

cd -
