#! /bin/bash

#$ -l rt_F=1
#$ -l h_rt=5:00:00
#$ -j y
#$ -cwd

source /etc/profile.d/modules.sh

module load python/3.6/3.6.5
module load cuda/10.0/10.0.130
module load cudnn/7.6/7.6.4
export NEW_VENV=${HOME}/B4new/deep
source ${NEW_VENV}/bin/activate
python ./run.py > "2019_11_17_wrn_40_2_search_with_random_erasing.txt"
