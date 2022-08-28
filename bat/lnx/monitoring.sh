#!/bin/bash
#
#SBATCH --job-name=Monitoring
#SBATCH --error=../outputs/Monitoring.err
#SBATCH --output=../outputs/Monitoring.txt
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
cp $HOME/LeleNet/dat/xls/SpeciesList.xlsx $TMP/dat/xls/SpeciesList.xlsx
source $HOME/venv/bin/activate
cp -r $HOME/LeleNet/dat/tls/Monitoring512 $TMP/dat/tls/03_2021
python3 $HOME/LeleNet/py3/LeleNet_trn.py "dl3" 16 300 -esp 40 -dd $TMP -ww 0.20 -op "adam" -lr 1e-4 -out "Monitoring"  -tbn "Monitoring"
deactivate
