import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

"""
<<Encoder>>
1. BFS_Encoder
    input_dim: 1+latent_dim
    output_dim: latent_dim
    return: [N,latent_dim]
"""
class BFS_Encoder(torch.nn.Module):
    def __init__(self,x_dim,latent_dim): 
        super().__init__()
        self.linear=nn.Linear(x_dim+latent_dim,latent_dim)
        self.relu=nn.ReLU()

    def forward(self,x,h):
        z=self.linear(torch.cat([x,h],dim=-1))
        return self.relu(z) # [N,latent_dim]

"""
<<Processor>>
1. MPNN_Processor
    aggr: max,min,avg
    return: [N,latent_dim] 
"""
class MPNN_Processor(MessagePassing):
    def __init__(self,latent_dim,e_dim,aggr='max'):
        super().__init__(aggr=aggr)
        self.M=nn.Linear(latent_dim+latent_dim+e_dim,latent_dim)
        self.U=nn.Linear(latent_dim+latent_dim,latent_dim)
        self.relu=nn.ReLU()

    def message(self,z_i,z_j,edge_attr): 
        m=self.M(torch.cat([z_i,z_j,edge_attr],dim=-1)) 
        return self.relu(m) # [E,latent_dim]

    def update(self,aggr_out,z):
        u=self.U(torch.cat([z,aggr_out],dim=-1)) 
        return self.relu(u) # [N,latent_dim]

    def forward(self,z,edge_index,edge_attr):
        return self.propagate(edge_index=edge_index,z=z,edge_attr=edge_attr)


"""
<<Decoder>>
1. BFS_Decoder
    input_dim: latent_dim+latent_dim
    output_dim: 1
2. BFS_Terminator
    input_dim: latent_dim
    output_dim: 1
    return: [1,1]
"""
class BFS_Decoder(torch.nn.Module):
    def __init__(self,latent_dim):
        super().__init__()
        self.linear=nn.Linear(latent_dim+latent_dim,1)

    def forward(self,z,h):
        output=self.linear(torch.cat([z,h],dim=-1))
        return output

class BFS_Terminator(torch.nn.Module):
    def __init__(self,latent_dim):
        super().__init__()
        self.linear=nn.Linear(latent_dim,1)

    def forward(self,h):
        h_mean=torch.mean(h,dim=0) # [latent_dim,]
        h_mean=h_mean.unsqueeze(0) # [1,latent_dim] 
        tau=self.linear(h_mean) # [1,1]
        return tau # [1,1]



"""
<<Model>>
1. NGAE_BFS
    output:
        h: [N,latent_dim]
        y: [N,1], logit
        tau: [1,1], logit
"""
class NGAE_BFS(torch.nn.Module):
    def __init__(self,x_dim,e_dim,latent_dim):
        super().__init__()
        self.encoder=BFS_Encoder(x_dim=x_dim,latent_dim=latent_dim)
        self.processor=MPNN_Processor(latent_dim=latent_dim,e_dim=e_dim)
        self.decoder=BFS_Decoder(latent_dim=latent_dim)
        self.terminator=BFS_Terminator(latent_dim=latent_dim)

    def forward(self,x,pre_h,edge_index,edge_attr):
        output={}
        z=self.encoder(x=x,h=pre_h)
        h=self.processor(z=z,edge_index=edge_index,edge_attr=edge_attr)
        y=self.decoder(z=z,h=h)
        tau=self.terminator(h=h)

        output['h']=h # [N,latent_dim]
        output['y']=y # [N,1], logit
        output['tau']=tau # [1,1], logit

        return output
