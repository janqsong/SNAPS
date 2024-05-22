import os
import torch
import random
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity

from gnn_cp.cp.cp_manager import CPManager
import gnn_cp.models.graph_models as graph_models
from gnn_cp.data.data_manager import GraphDataManager
from gnn_cp.models.model_manager import GraphModelManager


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Torch Graph Models are running on {device}")


def make_dataset_instances(
    dataset_manager: GraphDataManager,
    dataset_key,
    dataset,
    model_class_name,
    models_config,
    splits_config,
    models_cache_dir):

    instances = dataset_manager.load_splits(dataset_key, dataset, splits_config)
    
    print("Dataset Loaded Successfully!")
    print("====================================")

    print(f"Loading Models {model_class_name}")
    model_class = getattr(graph_models, model_class_name)

    for instance_idx, instance in enumerate(instances):
        model_manager = GraphModelManager(
            model_class_name=model_class_name,
            model_class=model_class,
            models_config=models_config,
            dataset=dataset,
            checkpoint_address=models_cache_dir,
            model_name=f"{dataset_key}-ins{instance_idx}-{model_class.__name__}")
        model_manager.load_model(dataset,instance)
        
        model_manager.model = model_manager.model.to(device)
        y_pred = model_manager.predict(
            dataset, test_idx=instance['test_idx'], return_embeddings=False
        )
        y_embeddings = model_manager.predict(
            dataset, return_embeddings=True
        )
        accuracy = accuracy_score(
            y_true=dataset.y[instance['test_idx']].cpu().numpy(),
            y_pred=y_pred.cpu().numpy(),
        )
        instance.update({"model": model_manager, "accuracy": accuracy, "embeddings": y_embeddings})
    print(
        f"Accuracy: {np.mean([instance['accuracy'] for instance in instances])} ± {np.std([instance['accuracy'] for instance in instances])}"
    )

    return instances


def get_overall_cp_result(
    dataset_manager: GraphDataManager,
    dataset,
    instances,
    selected_coverage,
    dataset_name,
    edge_index_initial,
    adj_knn):

    tune_fraction, calib_fraction, notune_calib_fraction = compute_tune_calib_fraction(instances[0])
    
    cp_manager = CPManager(dataset=dataset,
                           coverage_val=selected_coverage,
                           tune_fraction=tune_fraction,
                           calib_fraction=calib_fraction,
                           notune_calib_fraction=notune_calib_fraction,
                           dataset_name=dataset_name,
                           edge_index=edge_index_initial,
                           test_idx=None,
                           adj_knn=adj_knn)
    
    instance_results = []
    for instance in instances:
        embeddings = instance["embeddings"]
        tune_idx, test_idx = tune_truetest_split(dataset_manager, instance["test_idx"], dataset, tune_fraction)
        
        cp_keys = ['APS']
        cp_keys.append('DAPS-APS')
        cp_keys.append('SNAPS-APS')
        cp_keys.append('TSNAPS-APS')
        
        res = cp_manager.get_all_cp_results(cp_keys, embeddings, tune_idx, test_idx, instance["test_idx"])
        
        instance_results.append(res['mean'])
    instance_results = pd.concat(instance_results, axis=0, keys=[idx for idx in range(len(instances))])
    instance_mean_results = instance_results.reset_index().rename(
        columns={"level_0": "instance", "level_1": "method"})
    return instance_mean_results


def compute_tune_calib_fraction(instance, max_calib_num=1000):
    tune_num = min(instance["train_idx"].shape[0], 500)
    tune_fraction = tune_num / instance["test_idx"].shape[0]
    
    calib_num = min(max_calib_num, int(instance["test_idx"].shape[0] / 2))
    calib_fraction = (calib_num - tune_num) / (instance["test_idx"].shape[0] - tune_num)
    
    notune_calib_num = min(max_calib_num, int(instance["test_idx"].shape[0] / 2))
    notune_calib_fraction = notune_calib_num / instance["test_idx"].shape[0]
    
    return tune_fraction, calib_fraction, notune_calib_fraction

def tune_truetest_split(dataset_manager, test_idx, dataset, tuning_fraction):
    te_idx, tu_idx, _, _ = dataset_manager.train_test_split(test_idx, dataset.y[test_idx], training_fraction=tuning_fraction)
    return te_idx, tu_idx


def print_results(models_results, model_classes):
    models_results = pd.concat(models_results, axis=0, keys=model_classes)
    result = models_results.reset_index().rename(columns={"level_0": "model"})
    
    average_result = result.groupby(
        ["model","method"], sort=False).mean().reset_index().drop(columns=["level_1", "instance"])
    print(average_result.to_markdown())


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def compute_adj_knn(dataset_name, features, k=20):
    if not os.path.exists('./results/adj_knn/{}_adj_knn_{}'.format(dataset_name, k)):
        features = np.copy(features.cpu())
        features[features != 0] = 1
        sims = cosine_similarity(features)
        sims[(np.arange(len(sims)), np.arange(len(sims)))] = 0
        
        for i in range(len(sims)):
            indices_argsort = np.argsort(sims[i])
            sims[i, indices_argsort[:-k]] = 0
        
        A_feat = sp.coo_matrix(sims)
        row_sum = np.array(A_feat.sum(1))
        d_inv = np.power(row_sum, -1.0).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        A_feat = d_mat_inv.dot(A_feat).tocoo()
        
        adj_knn_st = sparse_mx_to_torch_sparse_tensor(A_feat).float()
        torch.save(adj_knn_st, './results/adj_knn/{}_adj_knn_{}.pt'.format(dataset_name, k))
    else:
        adj_knn_st = torch.load('./results/adj_knn/{}_adj_knn_{}.pt'.format(dataset_name, k))
    
    return adj_knn_st


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)