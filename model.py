from torch_geometric.nn import GCNConv, Linear
from torch_geometric.nn import global_mean_pool
from torch.nn import Module
from torch import manual_seed
import torch.nn.functional as F


class GCN(Module):
    def __init__(self, hidden_channels, dataset_num_node_features):
        super(GCN, self).__init__()
        # manual_seed(123)
        self.conv1 = GCNConv(dataset_num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels//2)
        self.conv3 = GCNConv(hidden_channels//2, hidden_channels//4)
        self.lin = Linear(hidden_channels//4, 2)  # 必须得写hidden_channels//4而不是hidden_channels/4
        # 必须要是整除符号

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x
