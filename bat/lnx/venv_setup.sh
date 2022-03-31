#!/bin/bash
python3 -m venv $HOME/venv
module purge
module load devel/cuda/11.4
source $HOME/venv/bin/activate
pip3 install -r requirements.txt
deactivate
