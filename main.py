import os
import yaml

from torch_sparse import SparseTensor

from utils import *
from gnn_cp.data.data_manager import GraphDataManager


#################################Config Setting################################

config_file_path = "./configs/config.yaml"
results_dir = "./results"
data_dir = "./datasets"

models_cache_dir = os.path.join(results_dir, "models")
if not os.path.exists(models_cache_dir):
    os.makedirs(models_cache_dir)
splits_dir = os.path.join(results_dir, "splits")
if not os.path.exists(splits_dir):
    os.makedirs(splits_dir)

with open(config_file_path, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

splits_config = config.get("baseline", {}).get("general_dataset_config", {})
models_config = config.get("baseline", {}).get("models", {})
model_classes = list(models_config.keys())

set_seed(42)
selected_coverage = 0.95
#################################Config Setting################################

dataset_manager = GraphDataManager(data_dir, splits_dir)
dataset_names = ['cora_ml', 'pubmed', 'citeseer', 'cora_full', 'coauthor_cs', 'coauthor_physics', 'amazon_computers', 'amazon_photo']
model_names = ['GCN', 'GAT', 'APPNP', 'MLP']

for selected_dataset in dataset_names:
    dataset = dataset_manager.get_dataset_from_key(selected_dataset)[0]
    edge_index_initial = dataset.edge_index.clone()
    dataset.edge_index = SparseTensor(row=dataset.edge_index[0], 
                                      col=dataset.edge_index[1], 
                                      value=torch.ones(dataset.edge_index.shape[1]), 
                                      sparse_sizes=(dataset.x.shape[0], dataset.x.shape[0]))
    print(selected_dataset, dataset)
    adj_knn = compute_adj_knn(selected_dataset, dataset.x, k=20)
    
    models_results = []
    for selected_model in model_names:
        instances = make_dataset_instances(dataset_manager, selected_dataset, dataset, selected_model, 
                                    models_config, splits_config, models_cache_dir)
        instance_mean_results = get_overall_cp_result(dataset_manager, dataset, instances, 
                                    selected_coverage, selected_dataset, edge_index_initial, adj_knn)
        models_results.append(instance_mean_results)
    print_results(models_results, model_names)