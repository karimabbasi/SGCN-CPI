# flag dataset is designed to choose the dataset
# set 0 for Davis dataset
# set 1 for KIBA dataset
# set 2 for BindingDB dataset

flag_dataset = 0

# threshold is designed to convert continious binding affinity to discrete (active or inactive)
# set 7 for Davis dataset
# set 12.1 for KIBA dataset
# set 7.6 for BindingDB dataset

threshold = 7

learning_rate_feat_model = 0.001

learning_rate_GCN = 0.00001

batch_size_test = 1

batch_size = 100

num_neighbours_test = 1000

smiles_max_len = 100
protein_max_len = 1000

smiles_dict_len = 64
protein_dict_len = 25

smiles_filter_lengths = 4
protein_filter_lengths = 8
num_filters = 32

embedding_size = 128

num_epoch_GCN = 20
num_epoch_GCN_inference = 20