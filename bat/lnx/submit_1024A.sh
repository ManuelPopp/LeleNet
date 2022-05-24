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
mkdir -p $TMP/dat/tls
cp -r $HOME/LeleNet/dat/tls/1024_best $TMP/dat/tls/03_2021
source $HOME/venv/bin/activate
python3 $HOME/LeleNet/py3/LeleNet_trn.py "unet" 4 45 -dd $TMP -ww 0.25 -op "rms" -dr 0.5 -lr 1e-4 -out "UNet025" -tbn "UNet025"
python3 $HOME/LeleNet/py3/LeleNet_trn.py "unet" 4 45 -dd $TMP -ww 0.50 -op "rms" -dr 0.5 -lr 1e-4 -out "UNet050" -tbn "UNet050"
python3 $HOME/LeleNet/py3/LeleNet_trn.py "unet" 4 45 -dd $TMP -ww 1.50 -op "rms" -dr 0.5 -lr 1e-4 -out "UNet150" -tbn "UNet150"
#python3 $HOME/LeleNet/py3/XX_Testing_and_visualisation.py
deactivate
