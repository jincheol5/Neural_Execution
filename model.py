import torch
from torch_geometric.nn import GATConv

class Encoder(torch.nn.Module):
  def __init__(self, in_dim, out_dim):
    super().__init__()
    self.linear = torch.nn.Linear(in_dim, out_dim)
    self.relu = torch.nn.ReLU()

  def forward(self, x):
    bfs_label = x.bfs_label.unsqueeze(1)
    input = torch.cat([bfs_label, x.h_bfs], dim=1)
    return self.relu(self.linear(input))


class Decoder(torch.nn.Module):
  def __init__(self, in_dim, out_dim):
    super().__init__()
    self.linear = torch.nn.Linear(in_dim, out_dim)

  def forward(self, z, h):
    input = torch.cat([z, h], dim=1)
    return self.linear(input) 


class Termination(torch.nn.Module):
  def __init__(self, in_dim, out_dim):
    super().__init__()
    self.linear = torch.nn.Linear(in_dim, out_dim)

  def forward(self, x):
    mean = torch.mean(x, dim=0)
    input = torch.cat((x, mean.unsqueeze(0)), dim=0)
    out = self.linear(input)
    return out[-1]

class GATProcessor(torch.nn.Module):
  def __init__(self, in_channels, out_channels, edge_dim, heads=1, dropout=0.0): #add_self_loops set to True by default, but what happens if there are already self loops?
    super().__init__()
    self.conv1 = GATConv(in_channels, out_channels, heads=heads, edge_dim=edge_dim, dropout=dropout)

  def forward(self, x, edge_index, edge_attr):
    x = self.conv1(x, edge_index, edge_attr)
    return x