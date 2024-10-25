import torch
import torch.nn as nn

class BFS_Encoder(torch.nn.Module):
  def __init__(self, input_dim, hidden_dim):
    super().__init__()
    self.linear=nn.Linear(input_dim+hidden_dim,hidden_dim)
    self.relu=nn.ReLU()

  def forward(self, x,h):
    z=self.linear(torch.cat([x,h],dim=1))
    return self.relu(z)

class BFS_Decoder(torch.nn.Module):
  def __init__(self, in_dim, out_dim):
    super().__init__()

  def forward(self, z, h):
    pass

class BFS_Terminator(torch.nn.Module):
  def __init__(self, in_dim, out_dim):
    super().__init__()

  def forward(self, x):
    pass

