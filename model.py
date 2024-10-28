import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

class BFS_Encoder(torch.nn.Module):
    def __init__(self, hidden_dim): 
        super().__init__()
        self.linear=nn.Linear(1+hidden_dim,hidden_dim)
        self.relu=nn.ReLU()

    def forward(self, x,h):
        z=self.linear(torch.cat([x,h],dim=-1))
        return self.relu(z)

class BFS_Decoder(torch.nn.Module):
    def __init__(self,hidden_dim):
        super().__init__()
        self.linear=nn.Linear(hidden_dim+hidden_dim,1)
        self.sigmoid=torch.nn.Sigmoid()

    def forward(self, z, h):
        output=self.linear(torch.cat([z,h],dim=-1))
        return self.sigmoid(output)

class BFS_Terminator(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.linear=nn.Linear(hidden_dim+hidden_dim, 1)

    def forward(self, h):
        h_mean=torch.mean(h,dim=0)
        output=self.linear(torch.cat([h,h_mean],dim=-1))
        output_mean=output.mean() # output_mean=(1,1)
        return output_mean

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
        ter=self.terminator(h=h)

        output['h']=h # (num_nodes,hidden_dim)
        output['y']=y # (num_nodes,1)
        output['ter']=ter # (1,1)

        return output