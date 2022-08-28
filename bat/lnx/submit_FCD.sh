#!/bin/bash
#
#SBATCH --job-name=LeleFCD
#SBATCH --error=../outputs/LeleNetF.err
#SBATCH --output=../outputs/LeleNetF.txt
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
#cp -r $HOME/LeleNet/dat/tls/BX_1024_21 $TMP/dat/tls/03_2021
#python3 $HOME/LeleNet/py3/LeleNet_trn.py "fcd" 1 300 -esp 40 -dd $TMP -ww 0.005 -op "adam" -lr 1e-4 -lrd 0.995 -out "F1024" -tbn "F1024"
#echo "1024 px run finished"
#rm -r $TMP/dat/tls/03_2021
cp -r $HOME/LeleNet/dat/tls/BX_512_21 $TMP/dat/tls/03_2021
python3 $HOME/LeleNet/py3/LeleNet_trn.py "fcd" 4 300 -esp 40 -dd $TMP -ww 0.005 -op "adam" -lr 1e-4 -lrd 0.995 -out "F512" -tbn "F512" -r "t"
echo "512 px run finished"
#rm -r $TMP/dat/tls/03_2021
#cp -r $HOME/LeleNet/dat/tls/BX_256_21 $TMP/dat/tls/03_2021
#python3 $HOME/LeleNet/py3/LeleNet_trn.py "fcd" 16 300 -esp 40 -dd $TMP -ww 0.005 -op "adam" -lr 1e-4 -lrd 0.995 -out "F256" -tbn "F256"
#echo "256 px run finished"
deactivate
