#!/bin/bash
#
#SBATCH --job-name=LeleUNet
#SBATCH --error=../outputs/LeleNetU.err
#SBATCH --output=../outputs/LeleNetU.txt
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
source $HOME/venv/bin/activate
cp $HOME/LeleNet/dat/xls/SpeciesList.xlsx $TMP/dat/xls/SpeciesList.xlsx
#cp -r $HOME/LeleNet/dat/tls/BX_1024_21 $TMP/dat/tls/03_2021
#python3 $HOME/LeleNet/py3/LeleNet_trn.py "unet" 4 300 -esp 150 -dd $TMP -ww 0.10 -op "adam" -lr 1e-4 -lrd 1.0 -out "U1024" -tbn "U1024"
#echo "1024 px run finished"
#rm -r $TMP/dat/tls/03_2021
cp -r $HOME/LeleNet/dat/tls/BX_512_21 $TMP/dat/tls/03_2021
python3 $HOME/LeleNet/py3/LeleNet_trn.py "unet" 16 300 -esp 150 -dd $TMP -ww 0.10 -op "adam" -lr 1e-4 -lrd 1.0 -out "U512" -tbn "U512"
echo "512 px run finished"
rm -r $TMP/dat/tls/03_2021
cp -r $HOME/LeleNet/dat/tls/BX_256_21 $TMP/dat/tls/03_2021
python3 $HOME/LeleNet/py3/LeleNet_trn.py "unet" 32 300 -esp 40 -dd $TMP -ww 0.10 -op "adam" -lr 1e-4 -lrd 1.0 -out "U256" -tbn "U256"
echo "256 px run finished"
deactivate
