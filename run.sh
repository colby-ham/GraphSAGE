# Generate new GraphSAGE format files
python convert_kgat_to_graphsage.py

# Then generate random walks
python graphsage/utils.py ./example_data/last-fm-G.json ./example_data/last-fm-walks.txt

# Then generate unsupervised embedding learning
#python -m graphsage.unsupervised_train --train_prefix ./example_data/last-fm --model graphsage_mean --epochs 10 --max_total_steps 1000 --validate_iter 10 --base_log_dir results --identity_dim 128
