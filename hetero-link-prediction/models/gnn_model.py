import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, to_hetero

class GNNEncoder(torch.nn.Module):
    """
    A homogeneous node encoder.
    This module will be converted dynamically to a heterogeneous graph model using `to_hetero`.
    """
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        # Lazy initialization via (-1, -1) handles automatically detecting input feature dimensions
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class EdgeDecoder(torch.nn.Module):
    """
    Predicts edges between given nodes using the representations of standard pair sources.
    """
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = torch.nn.Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, 1)

    def forward(self, z_dict, edge_label_index, edge_type):
        src_type, rel_type, dst_type = edge_type
        
        row, col = edge_label_index
        # Concatenate source node embeddings and destination node embeddings
        z = torch.cat([z_dict[src_type][row], z_dict[dst_type][col]], dim=-1)
        
        z = self.lin1(z)
        z = F.relu(z)
        z = self.lin2(z)
        return z.view(-1)

class HeteroLinkPredictionModel(torch.nn.Module):
    """
    Combined Encoder-Decoder model structured specifically for link prediction on 
    heterogeneous graphs. By using this modularity, metric learning modifications 
    can easily be swapped in place of the base components.
    """
    def __init__(self, hidden_channels, metadata):
        super().__init__()
        # PyG's to_hetero automatically adapts homogeneous message passing for bipartite/multi-partite node types
        self.encoder = to_hetero(GNNEncoder(hidden_channels, hidden_channels), metadata, aggr='sum')
        self.decoder = EdgeDecoder(hidden_channels)

    def forward(self, x_dict, edge_index_dict, edge_label_index, edge_type):
        # 1. Obtain heterogeneous node representations
        z_dict = self.encoder(x_dict, edge_index_dict)
        # 2. Predict link scores specifically for target edge type using dot product or MLP
        return self.decoder(z_dict, edge_label_index, edge_type)
