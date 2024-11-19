import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

### BFS
class BFS_Encoder(torch.nn.Module):
    def __init__(self, hidden_dim): 
        super().__init__()
        self.linear=nn.Linear(1+hidden_dim,hidden_dim)
        self.relu=nn.ReLU()

    def forward(self, x,h):
        z=self.linear(torch.cat([x,h],dim=-1)) # output=(N,hidden_dim)
        return self.relu(z)

class BFS_Decoder(torch.nn.Module):
    def __init__(self,hidden_dim):
        super().__init__()
        self.linear=nn.Linear(hidden_dim+hidden_dim,1)
        self.sigmoid=nn.Sigmoid()

    def forward(self, z, h):
        output=self.linear(torch.cat([z,h],dim=-1)) # output=(N,1)
        return self.sigmoid(output)

class BFS_Terminator(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.linear=nn.Linear(hidden_dim+hidden_dim, 1)
        self.sigmoid=nn.Sigmoid()

    def forward(self, h):
        N=h.size(0)
        h_mean=torch.mean(h,dim=0) # (hidden_feature,)
        h_mean=h_mean.unsqueeze(0) # (1,hidden_feature)
        h_mean=h_mean.expand(N,-1) # (n,hidden_feature)
        output=self.linear(torch.cat([h,h_mean],dim=-1)) # (N,1)
        output_mean = torch.mean(output, dim=0, keepdim=True)  # output_mean=(1, 1)
        return self.sigmoid(output_mean)

### Bellman-Ford
class BF_Encoder(torch.nn.Module):
    def __init__(self, hidden_dim): 
        super().__init__()
        self.linear=nn.Linear(1+hidden_dim,hidden_dim)
        self.relu=nn.ReLU()

    def forward(self, x,h):
        z=self.linear(torch.cat([x,h],dim=-1)) # output=(N,hidden_dim)
        return self.relu(z)

class BF_Decoder(torch.nn.Module):
    def __init__(self,hidden_dim,max_N):
        super().__init__()
        self.predecessor_linear=nn.Linear(hidden_dim+hidden_dim,max_N)
        self.distance_linear=nn.Linear(hidden_dim+hidden_dim,1)
        
    def forward(self, z, h):
        predecessor_output=self.predecessor_linear(torch.cat([z,h],dim=-1)) # predecessor output=(N,max_N)
        distance_output=self.distance_linear(torch.cat([z,h],dim=-1)) # distance output=(N,1)
        return predecessor_output,distance_output

class BF_Distance_Decoder(torch.nn.Module):
    def __init__(self,hidden_dim):
        super().__init__()
        self.distance_linear=nn.Linear(hidden_dim+hidden_dim,1)
        
    def forward(self, z, h):
        distance_output=self.distance_linear(torch.cat([z,h],dim=-1)) # distance output=(N,1)
        return distance_output

class BF_Terminator(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.linear=nn.Linear(hidden_dim+hidden_dim, 1)
        self.sigmoid=nn.Sigmoid()

    def forward(self, h):
        N=h.size(0)
        h_mean=torch.mean(h,dim=0) # (hidden_feature,)
        h_mean=h_mean.unsqueeze(0) # (1,hidden_feature)
        h_mean=h_mean.expand(N,-1) # (N,hidden_feature)
        output=self.linear(torch.cat([h,h_mean],dim=-1)) # (N,1)
        output_mean = torch.mean(output, dim=0, keepdim=True)  # output_mean=(1, 1)  
        return self.sigmoid(output_mean)


### Processor
class MPNN_Processor(MessagePassing):
    def __init__(self,hidden_dim):
        super().__init__(aggr="max")
        self.M=nn.Linear(hidden_dim+hidden_dim+1,hidden_dim)
        self.U=nn.Linear(hidden_dim+hidden_dim,hidden_dim)
        self.relu=nn.ReLU()

    def message(self,z_i,z_j,edge_attr): 
        # M: 이웃 노드들로부터 메시지 생성
        # 하나의 edge 당 message() 한 번씩 호출
        # propagate()에 처음 전달된 모든 인수를 취할 수 있다
        # z_i,z_j는 propagate() 에서 자동으로 매핑하여 전달
        # z_i,z_j=(num_edges,hidden_dim)
        # edge_attr=(num_edges,1)
        m=self.M(torch.cat([z_i,z_j,edge_attr],dim=-1))
        return self.relu(m)

    def update(self,aggr_out,z):
        # U: 노드 임베딩 업데이트
        # aggregate 출력을 첫 번째 인수로 받고 propagate()에 처음 전달된 모든 인수를 받음
        u=self.U(torch.cat([z,aggr_out],dim=-1))
        return self.relu(u)

    def forward(self, z,edge_index,edge_attr):
        # propagate()는 MessagePassing 클래스에 기본 구현 되어 있다
        # 1. message()
        # 2. aggregate() => aggr 함수 수행 (e.g., max)
        # 3. update()

        return self.propagate(edge_index=edge_index,z=z,edge_attr=edge_attr)

### Models
class BFS_Neural_Execution(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.encoder=BFS_Encoder(hidden_dim)
        self.processor=MPNN_Processor(hidden_dim)
        self.decoder=BFS_Decoder(hidden_dim)
        self.terminator=BFS_Terminator(hidden_dim)

    def forward(self, x,pre_h,edge_index,edge_attr):
        output={}
        z=self.encoder(x=x,h=pre_h)
        h=self.processor(z=z,edge_index=edge_index,edge_attr=edge_attr)
        y=self.decoder(z=z,h=h)
        tau=self.terminator(h=h)

        output['h']=h # h=(N,hidden_dim)
        output['y']=y # y=(N,1)
        output['tau']=tau # tau=(1,1)

        return output

class BF_Distance_Neural_Execution(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.encoder=BF_Encoder(hidden_dim)
        self.processor=MPNN_Processor(hidden_dim)
        self.decoder=BF_Distance_Decoder(hidden_dim)
        self.terminator=BF_Terminator(hidden_dim)

    def forward(self, x,pre_h,edge_index,edge_attr):
        output={}
        z=self.encoder(x=x,h=pre_h)
        h=self.processor(z=z,edge_index=edge_index,edge_attr=edge_attr)
        dist=self.decoder(z=z,h=h)
        tau=self.terminator(h=h)

        output['h']=h # h=(N,hidden_dim)
        output['dist']=dist # dist=(N,1)
        output['tau']=tau # tau=(1,1)

        return output

class BFS_BF_Distance_Neural_Execution(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.bfs_encoder=BF_Encoder(hidden_dim)
        self.bfs_decoder=BF_Distance_Decoder(hidden_dim)
        self.bfs_terminator=BF_Terminator(hidden_dim)
        self.bf_encoder=BF_Encoder(hidden_dim)
        self.bf_decoder=BF_Distance_Decoder(hidden_dim)
        self.bf_terminator=BF_Terminator(hidden_dim)
        self.processor=MPNN_Processor(hidden_dim)

    def forward(self, bfs_x,bf_x,bfs_pre_h,bf_pre_h,edge_index,edge_attr):
        output={}
        bfs_z=self.bfs_encoder(x=bfs_x,h=bfs_pre_h)
        bf_z=self.bf_encoder(x=bf_x,h=bf_pre_h)

        bfs_h=self.processor(z=bfs_z,edge_index=edge_index,edge_attr=edge_attr)
        bf_h=self.processor(z=bf_z,edge_index=edge_index,edge_attr=edge_attr)

        y=self.bfs_decoder(z=bfs_z,h=bfs_h)
        dist=self.bf_decoder(z=bf_z,h=bf_h)

        bfs_tau=self.bfs_terminator(h=bfs_h)
        bf_tau=self.bf_terminator(h=bf_h)

        output['bfs_h']=bfs_h # bfs_h=(N,hidden_dim)
        output['bf_h']=bfs_h # bf_h=(N,hidden_dim)
        output['y']=y # y=(N,1)
        output['dist']=dist # dist=(N,1)
        output['bfs_tau']=bfs_tau # bfs_tau=(1,1)
        output['bf_tau']=bf_tau # bf_tau=(1,1)

        return output