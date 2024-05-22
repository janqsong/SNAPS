import os
import argparse
import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.utils import sort_edge_index
from torch_geometric.nn import GCNConv

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from gnn_cp.cp.cp_manager import CPManager
from gnn_cp.data.data_manager import GraphDataManager
from utils import set_seed, sparse_mx_to_torch_sparse_tensor


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=True))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


def train(model, data, train_idx, optimizer):
    model.train()

    optimizer.zero_grad()
    out = model(data.x, data.adj_t)
    # loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])
    loss = F.cross_entropy(out[train_idx], data.y.squeeze(1)[train_idx])
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
    parser = argparse.ArgumentParser(description='OGBN-Arxiv (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=10)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--runs', type=int, default=10)
    args = parser.parse_args()
    print(args)
    
    set_seed(42)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='./large_datasets',
                                     transform=T.ToSparseTensor())

    data = dataset[0]
    print("ogbn-arxiv", data)
    data.adj_t = data.adj_t.to_symmetric()
    data = data.to(device)

    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device)
    
    model = GCN(data.num_features, args.hidden_channels,
                    dataset.num_classes, args.num_layers,
                    args.dropout).to(device)
    evaluator = Evaluator(name='ogbn-arxiv')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if "ogbn_arxiv_gcn.model" in os.listdir('.'):
        model = torch.load('ogbn_arxiv_gcn.model')
    else:        
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, data, train_idx, optimizer)
            result = test(model, data, split_idx, evaluator)

            if epoch % args.log_steps == 0:
                train_acc, valid_acc, test_acc = result
                print(f'Epoch: {epoch:02d}, '
                        f'Loss: {loss:.4f}, '
                        f'Train: {100 * train_acc:.2f}%, '
                        f'Valid: {100 * valid_acc:.2f}% '
                        f'Test: {100 * test_acc:.2f}%')
        torch.save(model, 'ogbn_arxiv_gcn.model')
    
    result = test(model, data, split_idx, evaluator)
    print(result)
    
    model.eval()
    embeddings = model(data.x, data.adj_t)
    
    
    ####################################Conformal Prediction##################################
    print(split_idx['test'])
    adj_knn = compute_ogbn_adj_knn("ogbn_arxiv", data.x, split_idx['test'], k=20)
    
    data.y = data.y.squeeze(1)
    edge_index = sort_edge_index(PygNodePropPredDataset(name='ogbn-arxiv', root='./large_datasets')[0].edge_index).cuda()
    selected_coverage = 0.95
    
    tune_fraction = np.round(500 / (split_idx['test'].shape[0]), 10)
    calib_fraction = np.round(500 / (split_idx['test'].shape[0] - 500), 10)
    
    notune_calib_fraction = np.round(1000 / (split_idx['test'].shape[0]), 10)
    
    tune_idx, test_idx, _, _ = GraphDataManager.train_test_split(split_idx['test'], 
                                                                 dataset.y[split_idx['test']], training_fraction=tune_fraction)
    
    
    cp_manager = CPManager(dataset=data,
                           coverage_val=selected_coverage,
                           tune_fraction=tune_fraction,
                           calib_fraction=calib_fraction,
                           notune_calib_fraction=notune_calib_fraction,
                           dataset_name='ogbn_arxiv',
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
    num_nodes = features.shape[0]
    num_test_nodes = test_idx.shape[0]
    
    random_idx = torch.randint(low=0, high=num_nodes, size=(80000, ))
    same_indices = torch.transpose(torch.nonzero(torch.eq(test_idx[:, None], random_idx)), 0, 1)
    
    if not os.path.exists('./results/adj_knn/{}_adj_knn_{}.pt'.format(dataset_name, k)):
        if not os.path.exists('./results/adj_knn/{}_sims_{}.npz'.format(dataset_name, k)):
            test_features = np.copy(features[test_idx, :].cpu())
            random_features = np.copy(features[random_idx, :].cpu())
            
            sims = cosine_similarity(test_features, random_features)
            sims[tuple(same_indices)] = 0
            for i in range(len(sims)):
                indices_argsort = np.argsort(sims[i])
                sims[i, indices_argsort[: -k]] = 0
                if i % 1000 == 0:
                    print(i)
            A_feat = sp.csr_matrix(sims)
            sp.save_npz('./results/adj_knn/{}_sims_{}.npz'.format(dataset_name, k), A_feat)
        else:
            A_feat = sp.load_npz('./results/adj_knn/{}_sims_{}.npz'.format(dataset_name, k))
        
        print(A_feat)
        row_sum = np.array(A_feat.sum(1))
        d_inv = np.power(row_sum, -1.0).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        A_feat = d_mat_inv.dot(A_feat).tocoo()
        
        adj_knn_st = sparse_mx_to_torch_sparse_tensor(A_feat).float()
        adj_knn_st = adj_knn_st.coalesce()
        indices = adj_knn_st.indices()
        values = adj_knn_st.values()
        indices[0, :] = test_idx[indices[0, :]]
        indices[1, :] = random_idx[indices[1, :]]
        adj_knn = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes))
        print(adj_knn)
        torch.save(adj_knn, './results/adj_knn/{}_adj_knn_{}.pt'.format(dataset_name, k))
    else:
        adj_knn = torch.load('./results/adj_knn/{}_adj_knn_{}.pt'.format(dataset_name, k))
    
    return adj_knn
    
if __name__ == "__main__":
    main()