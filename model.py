import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

class BFS_Encoder(torch.nn.Module):
    def __init__(self, hidden_dim): # node feature=(batch_size,1) 
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

    def forward(self, z, h):
        return self.linear(torch.cat([z,h],dim=-1))

class BFS_Terminator(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.linear=nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        pass

class MPNN_Processor(MessagePassing):
    def __init__(self,hidden_dim):
        super().__init__(aggr="max")
        self.M=nn.Linaer(hidden_dim+hidden_dim+1,hidden_dim)
        self.U=nn.Linear(hidden_dim+hidden_dim,hidden_dim)
        self.relu=nn.ReLU()

    def message(self,z_i,z_j,edge_attr): 
        # M: 이웃 노드들로부터 메시지 생성
        # 하나의 edge 당 message() 한 번씩 호출
        # propagate()에 처음 전달된 모든 인수를 취할 수 있다
        # z_i,z_j는 propagate() 에서 자동으로 매핑하여 전달  
        return self.relu(self.M(torch.cat([z_i,z_j,edge_attr],dim=-1)))

    def update(self,aggr_out,z):
        # U: 노드 임베딩 업데이트
        # aggregate 출력을 첫 번째 인수로 받고 propagate()에 처음 전달된 모든 인수를 받음
        return self.relu(self.M(torch.cat([z,aggr_out],dim=-1)))

    def forward(self, z,edge_index,edge_attr):
        # propagate()는 MessagePassing 클래스에 기본 구현 되어 있다
        # 1. message()
        # 2. aggregate() => aggr 함수 수행 (e.g., max)
        # 3. update()
        return self.propagate(edge_index,z=z,edge_attr=edge_attr)