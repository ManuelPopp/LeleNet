#!/bin/bash
#
#SBATCH --job-name=LeleNetA
#SBATCH --error=../outputs/LeleNet1024A.err
#SBATCH --output=../outputs/LeleNet1024A.txt
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
python3 $HOME/LeleNet/py3/LeleNet_trn.py "fcd" 1 50 -yr "1024_3_class_EbLd" -ww 0.5 -op "adam"
#python3 $HOME/LeleNet/py3/XX_Testing_and_visualisation.py
deactivate