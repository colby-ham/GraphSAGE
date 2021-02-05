#!/bin/bash

#SBATCH --job-name=gsage-last_fm
#SBATCH --output=logs/%x-%j.out
#SBATCH -A st_graphs
#SBATCH -p tonga
#SBATCH -n 1
#SBATCH -t 47:59:00
#SBATCH --reservation=st_graphs
#SBATCH -A st_graphs

source /etc/profile.d/modules.sh
module purge
module load cuda/9.2.148 
#module load python/anaconda3.2019.3
module load python/anaconda3.2020.02
module load gcc/7.3.0
source /share/apps/anaconda3.2020.02/etc/profile.d/conda.sh
source activate graphsage
PATH=/bin:$PATH

repo_dir=/people/hamc649/recommendation/GraphSAGE
echo "repo_dir: "${repo_dir}
cd $repo_dir

dataset=last-fm

# Generate new GraphSAGE format files
python convert_kgat_to_graphsage.py $dataset

# Then generate random walks
python graphsage/utils.py ./example_data/${dataset}-G.json ./example_data/${dataset}-walks.txt

# Then generate unsupervised embedding learning
python -m graphsage.unsupervised_train --train_prefix ./example_data/$dataset --model graphsage_mean --epochs 10 --max_total_steps 1000 --validate_iter 10 --base_log_dir results --identity_dim 128
