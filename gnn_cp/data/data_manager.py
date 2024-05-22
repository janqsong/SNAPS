import os
import torch
import numpy as np
from importlib import import_module

import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.utils import subgraph


class RemoveNodes:
    """
    Remove all the nodes that belong to a class with a number of samples less than 60.
    """

    def __init__(self, min_samples_per_class=50):
        self.min_samples_per_class = min_samples_per_class

    def _relabel_classes(self, y):
        classes = torch.unique(y).tolist()
        new_labels = torch.arange(len(classes)).tolist()
        label_map = dict(zip(classes, new_labels))
        mp = np.arange(0, y.max() + 1)
        mp[list(label_map.keys())] = list(label_map.values())
        y = torch.from_numpy(mp[y])
        return y
    
    def __call__(self, data):
        _, num_samples_per_class = torch.unique(data.y, return_counts=True)

        nodes_mask = []
        for cls in data.y:
            if num_samples_per_class[cls.item()] < self.min_samples_per_class:
                nodes_mask.append(False)
            else:
                nodes_mask.append(True)
        nodes_mask = torch.BoolTensor(nodes_mask)

        edge_index, _ = subgraph(
            nodes_mask, data.edge_index, relabel_nodes=True, return_edge_mask=False
        )

        x = data.x[nodes_mask, :]

        # reindex y
        remaining_y = data.y[nodes_mask]
        y = self._relabel_classes(remaining_y)
        return Data(x=x, edge_index=edge_index, y=y)


class GraphDataManager(object):
    dataset_libs = {"torch_geo": "torch_geometric.datasets"}
    
    dataset_shortcuts = {
        "cora_ml": {
            "dataset_name": "Cora_ML",
            "dataset_superobject": "CitationFull",
            "lib_key": "torch_geo",
        },
        "pubmed": {
            "dataset_name": "PubMed",
            "dataset_superobject": "CitationFull",
            "lib_key": "torch_geo",
        },
        "citeseer": {
            "dataset_name": "CiteSeer",
            "dataset_superobject": "CitationFull",
            "lib_key": "torch_geo",
        },
        "dblp": {
            "dataset_name": "DBLP",
            "dataset_superobject": "CitationFull",
            "lib_key": "torch_geo",
        },
        "cora_full": {
            "dataset_name": "Cora",
            "dataset_superobject": "CitationFull",
            "lib_key": "torch_geo",
            "pre_transform": [
                "RemoveNodes",
            ],
        },
        "coauthor_cs": {
            "dataset_name": "CS",
            "dataset_superobject": "Coauthor",
            "lib_key": "torch_geo",
        },
        "coauthor_physics": {
            "dataset_name": "Physics",
            "dataset_superobject": "Coauthor",
            "lib_key": "torch_geo",
        },
        "amazon_computers": {
            "dataset_name": "Computers",
            "dataset_superobject": "Amazon",
            "lib_key": "torch_geo",
        },
        "amazon_photo": {
            "dataset_name": "Photo",
            "dataset_superobject": "Amazon",
            "lib_key": "torch_geo",
        }
    }
    
    def __init__(self, path=None, splits_path=None):
        if path == None:
            raise ValueError("You must enter a path for data storage!")
        self.path = path
        self.splits_path = splits_path
        
    def get_dataset_from_key(self, dataset_key):
        keywords = self.dataset_shortcuts.get(dataset_key)
        if keywords is None:
            raise KeyError("Dataset shortcut does not exist")
        return self.get_dataset(**keywords)
    
    def get_dataset(
        self,
        dataset_name,
        dataset_superobject,
        lib_key="torch_geo",
        pre_transform=None,
        root=None):
        loader_object = self.dataset_loader_object(
            dataset_superobject, self.dataset_libs.get(lib_key)
        )
        
        if pre_transform is not None:
            pre_transform = T.Compose([eval(t)() for t in pre_transform])

        root = os.path.join(self.path, root) if root is not None else self.path
        data = loader_object(root=root, name=dataset_name, pre_transform=pre_transform)
        
        return data
    
    def dataset_loader_object(self, dataset_superobject, library):
        dataset_loader_object = getattr(import_module(library), dataset_superobject)
        return dataset_loader_object
    
    def init_k_splits(self, data, n_train, n_val, k=10):
        result = [
            self.stratified_sampling_idx(data, n_train, n_val)
            for _ in range(k)
        ]
        return result
    
    @staticmethod
    def train_test_split(
        X=None,
        y=None,
        data=None,
        training_fraction=0.75,
        class_balanced=False,
        return_idx=False,
        train_nodes_per_class=None):
        if X is None and data is not None:
            X = data.x
            y = data.y
        training_idx = None
        test_idx = None
        if class_balanced:
            classes = y.unique()
            classes_idx_set = [
                (y == cls_val).nonzero(as_tuple=True)[0] for cls_val in classes
            ]
            shuffled_classes = [s[torch.randperm(s.shape[0])] for s in classes_idx_set]
            split_points = [
                int(training_fraction * s.shape[0]) for s in shuffled_classes
            ]
            if not train_nodes_per_class is None:
                split_points = [train_nodes_per_class for s in shuffled_classes]
            training_idxs = [
                s[: split_points[i_s]] for i_s, s in enumerate(shuffled_classes)
            ]
            test_idxs = [
                s[split_points[i_s] :] for i_s, s in enumerate(shuffled_classes)
            ]
            training_idx = torch.concat(training_idxs)
            test_idx = torch.concat(test_idxs)

            new_training_prem = torch.randperm(training_idx.shape[0])
            new_test_prem = torch.randperm(test_idx.shape[0])
            training_idx = training_idx[new_training_prem]
            test_idx = test_idx[new_test_prem]
        else:
            perm = torch.randperm(X.shape[0])

            split_index = int(round(training_fraction * X.shape[0]))
            training_idx = perm[:split_index]
            test_idx = perm[split_index:]

        if return_idx:
            return training_idx, test_idx
        return X[training_idx], X[test_idx], y[training_idx], y[test_idx]
    
    @staticmethod
    def stratified_sampling_idx(data, train_n, val_n):
        
        train_idx, val_test_idx = GraphDataManager.train_test_split(
            data=data,
            train_nodes_per_class=train_n,
            class_balanced=True,
            return_idx=True)
        
        val_idx, test_idx, _, _ = GraphDataManager.train_test_split(
            X=val_test_idx,
            y=data.y[val_test_idx],
            train_nodes_per_class=val_n,
            class_balanced=True)
        
        result_dict = {"train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx}
        return result_dict
    
    def save_splits(self, dataset_key, splits):
        for split_idx, split in enumerate(splits):
            file_group_name = f"{dataset_key}--{split_idx}"
            train_file_name = os.path.join(self.splits_path, f"{file_group_name}--train")
            val_file_name = os.path.join(self.splits_path, f"{file_group_name}--val")
            test_file_name = os.path.join(self.splits_path, f"{file_group_name}--test")

            torch.save(split["train_idx"], train_file_name)
            torch.save(split["val_idx"], val_file_name)
            torch.save(split["test_idx"], test_file_name)
    
    def load_splits(self, dataset_key, dataset, splits_config):
        split_files = [item
                       for item in os.listdir(self.splits_path)
                       if dataset_key + "--" in item]
        split_keys = sorted(list({item.split("--")[1] for item in split_files}))
        instances = []
        for key in split_keys:
            instances.append(
                {
                    "train_idx": torch.load(
                        os.path.join(self.splits_path, f"{dataset_key}--{key}--train")
                    ),
                    "val_idx": torch.load(
                        os.path.join(self.splits_path, f"{dataset_key}--{key}--val")
                    ),
                    "test_idx": torch.load(
                        os.path.join(self.splits_path, f"{dataset_key}--{key}--test")
                    ),
                }
            )
            
        if len(instances) == 0:
            print("===============")
            n_splits = splits_config.get("n_init_splits", 10)
            n_train = splits_config.get("n_train_labels", 20)
            n_val = splits_config.get("n_val_labels", 20)
            
            print("Creating splits")
            new_splits = self.init_k_splits(
                dataset, n_train, n_val, k=n_splits
            )
                
            self.save_splits(
                dataset_key=dataset_key, splits=new_splits
            )
            instances += new_splits
        

        return instances
    
    @staticmethod
    def random_string_tail():
        import random
        import string

        def randStr(chars=string.ascii_uppercase + string.digits, N=10):
            return "".join(random.choice(chars) for _ in range(N))

        return randStr(N=10)
    
    @staticmethod
    def get_num_classes(data):
        return data.y.max().item() + 1