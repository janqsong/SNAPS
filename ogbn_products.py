import os
import argparse
import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.utils import sort_edge_index

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from utils import set_seed, sparse_mx_to_torch_sparse_tensor
from gnn_cp.cp.cp_manager import CPManager
from gnn_cp.data.data_manager import GraphDataManager


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels, normalize=False))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, normalize=False))
        self.convs.append(
            GCNConv(hidden_channels, out_channels, normalize=False))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return torch.log_softmax(x, dim=-1)


def train(model, data, train_idx, optimizer):
    model.train()

    optimizer.zero_grad()
    out = model(data.x, data.adj_t)[train_idx]
    loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, data, split_idx, evaluator):
    model.eval()

    out = model(data.x, data.adj_t)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc


def main():
    parser = argparse.ArgumentParser(description='OGBN-Products (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_sage', action='store_true')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--runs', type=int, default=10)
    args = parser.parse_args()
    print(args)
    
    set_seed(0)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygNodePropPredDataset(name='ogbn-products', root='./large_datasets',
                                     transform=T.ToSparseTensor())
    data = dataset[0]
    print("ogbn-products", data)
    split_idx = dataset.get_idx_split()
    print(split_idx['train'].shape)
    print(split_idx['valid'].shape)
    print(split_idx['test'].shape)
    
    train_idx = split_idx['train'].to(device)

    model = GCN(data.num_features, args.hidden_channels,
                dataset.num_classes, args.num_layers,
                args.dropout).to(device)

    # Pre-compute GCN normalization.
    adj_t = data.adj_t.set_diag()
    deg = adj_t.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
    data.adj_t = adj_t

    data = data.to(device)

    evaluator = Evaluator(name='ogbn-products')


    if os.path.exists("./results/models/ogbn_products_gcn.model"):
        model = torch.load("./results/models/ogbn_products_gcn.model")
    else:
        model.reset_parameters()
        print("============Start===========")
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(1, 1 + args.epochs):
            print("train")
            loss = train(model, data, train_idx, optimizer)
            print("test")
            result = test(model, data, split_idx, evaluator)

            # if epoch % args.log_steps == 10:
            train_acc, valid_acc, test_acc = result
            print(f'Epoch: {epoch:02d}, '
                    f'Loss: {loss:.4f}, '
                    f'Train: {100 * train_acc:.2f}%, '
                    f'Valid: {100 * valid_acc:.2f}% '
                    f'Test: {100 * test_acc:.2f}%')
        torch.save(model, "./results/models/ogbn_products_gcn.model")
    
    result = test(model, data, split_idx, evaluator)
    print(result)
    
    model.eval()
    embeddings = model(data.x, data.adj_t).cpu()
    
    ####################################Conformal Prediction##################################
    print(split_idx['test'])
    adj_knn = compute_ogbn_adj_knn("ogbn-products", data.x, split_idx['test'], k=20)
    
    data.y = data.y.squeeze(1)
    edge_index = sort_edge_index(PygNodePropPredDataset(name='ogbn-products', root='./large_datasets')[0].edge_index).cuda()
    selected_coverage = 0.95
    
    tune_fraction = np.round(1000 / (split_idx['test'].shape[0]), 10)
    calib_fraction = np.round(1000 / (split_idx['test'].shape[0] - 1000), 10)
    
    notune_calib_fraction = np.round(2000 / (split_idx['test'].shape[0]), 10)
    
    tune_idx, test_idx, _, _ = GraphDataManager.train_test_split(split_idx['test'], 
                                                                 dataset.y[split_idx['test']], training_fraction=tune_fraction)
    
    print(f"{torch.cuda.memory_allocated()} bytes")
    cp_manager = CPManager(dataset=data,
                           coverage_val=selected_coverage,
                           tune_fraction=tune_fraction,
                           calib_fraction=calib_fraction,
                           notune_calib_fraction=notune_calib_fraction,
                           dataset_name='ogbn-products',
                           edge_index=edge_index,
                           test_idx=split_idx['test'],
                           adj_knn=adj_knn)

    aps_res = cp_manager.get_aps_result(embeddings, tune_idx, test_idx, split_idx['test'])
    print(aps_res.mean().drop("attempt"))

    daps_res = cp_manager.get_daps_result(cp_manager.base_scores['APS'], tune_idx, test_idx, split_idx['test'])
    print(daps_res.mean().drop("attempt"))
    
    snaps_res = cp_manager.get_snaps_result(cp_manager.base_scores['APS'], tune_idx, test_idx, split_idx['test'])
    print(snaps_res.mean().drop("attempt"))
    
    temp_snaps_res = cp_manager.get_temp_snaps_result(cp_manager.base_scores['APS'], tune_idx, test_idx, split_idx['test'])
    print(temp_snaps_res.mean().drop("attempt"))
    

def compute_ogbn_adj_knn(dataset_name, features, test_idx, k=20):
    if os.path.exists('./results/adj_knn/{}_adj_knn.pt'.format(dataset_name)):
        adj_knn = torch.load('./results/adj_knn/{}_adj_knn.pt'.format(dataset_name))
        return adj_knn
    adj_knn_final = None
    
    num_nodes = features.shape[0]
    chunk_test_idxs = torch.chunk(test_idx, chunks=int(test_idx.shape[0]/80000), dim=0)
    
    for idx, chunk_test_idx in enumerate(chunk_test_idxs):
        print(chunk_test_idx.shape)
        random_idx = torch.randint(low=0, high=num_nodes, size=(80000, ))
        same_indices = torch.transpose(torch.nonzero(torch.eq(chunk_test_idx[:, None], random_idx)), 0, 1)
        
        if os.path.exists('./results/adj_knn/{}_adj_knn_{}.pt'.format(dataset_name, idx)):
            adj_knn = torch.load('./results/adj_knn/{}_adj_knn_{}.pt'.format(dataset_name, idx))
        else:
            if not os.path.exists('./results/adj_knn/{}_sims_{}.npz'.format(dataset_name, idx)):
                test_features = np.copy(features[chunk_test_idx, :].cpu())
                random_features = np.copy(features[random_idx, :].cpu())
                
                sims = cosine_similarity(test_features, random_features)
                sims[tuple(same_indices)] = 0
                for i in range(len(sims)):
                    indices_argsort = np.argsort(sims[i])
                    sims[i, indices_argsort[: -k]] = 0
                    if i % 1000 == 0:
                        print(idx, i)
                A_feat = sp.csr_matrix(sims)
                sp.save_npz('./results/adj_knn/{}_sims_{}.npz'.format(dataset_name, idx), A_feat)
            else:
                A_feat = sp.load_npz('./results/adj_knn/{}_sims_{}.npz'.format(dataset_name, idx))
            
            row_sum = np.array(A_feat.sum(1))
            d_inv = np.power(row_sum, -1.0).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            A_feat = d_mat_inv.dot(A_feat).tocoo()
            
            adj_knn_st = sparse_mx_to_torch_sparse_tensor(A_feat).float()
            adj_knn_st = adj_knn_st.coalesce()
            indices = adj_knn_st.indices()
            values = adj_knn_st.values()
            indices[0, :] = chunk_test_idx[indices[0, :]]
            indices[1, :] = random_idx[indices[1, :]]
            adj_knn = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes))
            
            torch.save(adj_knn, './results/adj_knn/{}_adj_knn_{}.pt'.format(dataset_name, idx))
        if adj_knn_final is None:
            adj_knn_final = adj_knn
        else:
            adj_knn_final = adj_knn_final.coalesce()
            adj_knn = adj_knn.coalesce()
            indices_combined = torch.cat([adj_knn_final.indices(), adj_knn.indices()], dim=1)
            values_combined = torch.cat([adj_knn_final.values(), adj_knn.values()])
            adj_knn_final = torch.sparse_coo_tensor(indices_combined, values_combined, (num_nodes, num_nodes))
        print(adj_knn_final)
    print(adj_knn_final)
    torch.save(adj_knn_final, './results/adj_knn/{}_adj_knn.pt'.format(dataset_name))
    return adj_knn_final


if __name__ == "__main__":
    main()