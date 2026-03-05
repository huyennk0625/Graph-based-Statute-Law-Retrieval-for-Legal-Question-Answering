import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv

class GraphSAGEModel(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, in_dim)
        self.norm = nn.LayerNorm(in_dim)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = self.norm(h)
        return x + 0.05 * h