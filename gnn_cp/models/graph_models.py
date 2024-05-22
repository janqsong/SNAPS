import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, APPNP, SAGEConv, FAConv
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Torch Graph Models are running on {device}")


class GCN(torch.nn.Module):
    def __init__(self, n_features, n_hidden, n_classes, p_dropout):
        super().__init__()
        self.conv1 = GCNConv(n_features, n_hidden, cached=False,
                             normalize=True)
        self.conv2 = GCNConv(n_hidden, n_classes, cached=False,
                             normalize=True)
        self.p_dropout = p_dropout

    def forward(self, X, edge_weight=None):
        x, edge_index = X.x, X.edge_index
        # x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x
    

class GAT(torch.nn.Module):
    def __init__(self, n_features, n_hidden, n_classes, n_heads, p_dropout=0.6, p_dropout_attention=0.3):
        super().__init__()
        self.conv1 = GATConv(n_features, n_hidden, n_heads, dropout=p_dropout_attention)
        # On the Pubmed dataset, use `heads` output heads in `conv2`.
        self.conv2 = GATConv(n_hidden * n_heads, n_classes, n_heads=1,
                             concat=False)
        self.p_dropout = p_dropout
        self.p_dropout_attention = p_dropout_attention

    def forward(self, X):
        x, edge_index = X.x, X.edge_index
        # x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x, edge_weight = self.conv2(x, edge_index, return_attention_weights=True)
        return x
    

class APPNPNet(nn.Module):
    def __init__(self, n_features, n_hidden, n_classes, k_hops, alpha, p_dropout= 0.5):
        super(APPNPNet, self).__init__()
        self.lin1 = nn.Linear(n_features, n_hidden, bias=False)
        self.lin2 = nn.Linear(n_hidden, n_classes, bias=False)
        self.prop = APPNP(K = k_hops, alpha = alpha)
        self.p_dropout = p_dropout

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, X):
        node_attr, edge_index = X.x, X.edge_index

        hidden = F.relu(self.lin1(node_attr))
        hidden = F.dropout(hidden, p=self.p_dropout, training=self.training)
        out = self.lin2(hidden)
        out = self.prop(out, edge_index)
        return F.log_softmax(out, dim=1)

class MLP(nn.Module):
    def __init__(self, n_features, n_hidden, n_classes, p_dropout= 0.5):
        super(MLP, self).__init__()
        self.lin1 = nn.Linear(n_features, n_hidden)
        self.lin2 = nn.Linear(n_hidden, n_classes)

        self.p_dropout = p_dropout

    def forward(self, X):
        node_attr, edge_index = X.x, X.edge_index
        hidden = F.relu(self.lin1(node_attr))
        hidden = F.dropout(hidden, p=self.p_dropout, training=self.training)
        out = self.lin2(hidden)
        return out