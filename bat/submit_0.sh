#!/bin/bash
#
#SBATCH --job-name=LeleNet0
#SBATCH --error=../outputs/LeleNet0.err
#SBATCH --output=../outputs/LeleNet0.txt
#
#SBATCH --partition=gpu_8
#SBATCH --gres=gpu:4
#
#SBATCH --cpus-per-task=40
#SBATCH --time=24:00:00
#SBATCH --mem=360000
#
module purge
module load devel/cuda/11.4
source $HOME/venv/bin/activate
python3 $HOME/LeleNet/py3/LeleNet_trn.py "dl3" 5 50 -yr "felix" -ww 1.0 -op "adam" -lc 1
#python3 $HOME/LeleNet/py3/XX_Testing_and_visualisation.py
deactivate
