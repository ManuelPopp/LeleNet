#!/bin/bash
#
#SBATCH --job-name=LeleDL3
#SBATCH --error=../outputs/LeleNetD.err
#SBATCH --output=../outputs/LeleNetD.txt
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
#python3 $HOME/LeleNet/py3/LeleNet_trn.py "dl3" 4 300 -esp 40 -dd $TMP -ww 0.20 -op "adam" -lr 1e-4 -lrd 0.995 -out "D1024" -tbn "D1024"
#echo "1024 px run finished"
#rm -r $TMP/dat/tls/03_2021
cp -r $HOME/LeleNet/dat/tls/BX_512_21 $TMP/dat/tls/03_2021
cp -r $HOME/LeleNet/dat/tls/BX_512_22 $TMP/dat/tls/03_2021
python3 $HOME/LeleNet/py3/LeleNet_trn.py "dl3" 16 30 -esp 40 -dd $TMP -ww 0.20 -op "adam" -lr 1e-4 -lrd 0.995 -out "D512" -tbn "D512"
echo "512 px run finished"
rm -r $TMP/dat/tls/03_2021
cp -r $HOME/LeleNet/dat/tls/BX_256_21 $TMP/dat/tls/03_2021
python3 $HOME/LeleNet/py3/LeleNet_trn.py "dl3" 32 300 -esp 40 -dd $TMP -ww 0.20 -op "adam" -lr 1e-4 -lrd 0.995 -out "D256" -tbn "D256"
echo "256 px run finished"
deactivate
