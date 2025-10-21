#!/bin/bash

set -e
echo "This will take a while, make some coffee, sit back, and relax :)"
ENV_NAME="lmgame"
echo "STEP 1: Checking for Conda environment '$ENV_NAME'."

if ! conda env list | grep -q "$ENV_NAME"; then
  echo "Creating enviornment. Expect current_repodata to fail, repodata will work."
  conda create -n "$ENV_NAME" python=3.10 -y
else
  echo "Environment '$ENV_NAME' already exists."
fi

echo "STEP 2: Installing libraries to enviornment"
conda run -n "$ENV_NAME" pip install -v -e.

echo "STEP 3: Installing libGl, used for graphics interfaces."
conda install -n "$ENV_NAME" -c conda-forge libgl -y

echo "STEP 4: Correcting Numpy (We use Numba, which is a deprecated feature)"
conda run -n "$ENV_NAME" pip install -v numpy==1.26


echo "Enviornment is installed. Run 'conda activate lmgame' in your bash terminal before running anything"