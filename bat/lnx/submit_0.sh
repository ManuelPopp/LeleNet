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
mkdir -p $TMP/dat/{tls,xls}
cp -r $HOME/LeleNet/dat/tls/512_n_class $TMP/dat/tls/03_2021
cp $HOME/LeleNet/dat/xls/SpeciesList.xlsx $TMP/dat/xls/SpeciesList.xlsx
source $HOME/venv/bin/activate
python3 $HOME/LeleNet/py3/LeleNet_trn.py "unet" 8 45 -dd $TMP -ww 0.25 -op "rms" -dr 0.0 -lr 1e-4 -out "UNet0_025" -tbn "UNet0025"
python3 $HOME/LeleNet/py3/LeleNet_trn.py "unet" 8 45 -dd $TMP -ww 0.50 -op "rms" -dr 0.0 -lr 1e-4 -out "UNet0_050" -tbn "UNet0050"
python3 $HOME/LeleNet/py3/LeleNet_trn.py "unet" 8 45 -dd $TMP -ww 1.50 -op "rms" -dr 0.0 -lr 1e-4 -out "UNet0_150" -tbn "UNet0150"
#python3 $HOME/LeleNet/py3/XX_Testing_and_visualisation.py
deactivate
