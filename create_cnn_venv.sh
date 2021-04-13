#!/usr/bin/env bash

VENVNAME=cnn_venv 

python3 -m venv $VENVNAME
source $VENVNAME/bin/activate
pip install --upgrade pip

# problems when installing from requirements.txt
pip install ipython
pip install jupyter
pip install matplotlib
pip install numpy
pip install sklearn
pip install tensorflow


python -m ipykernel install --user --name=$VENVNAME

test -f cnn_requirements.txt && pip install cnn_requirements.txt

deactivate
echo "build $VENVNAME"

