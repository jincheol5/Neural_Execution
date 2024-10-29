import torch
from torch_geometric.data import Data

print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)

# 간단한 PyG 데이터 생성
edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long).cuda()
x = torch.tensor([[1], [2]], dtype=torch.float).cuda()
data = Data(x=x, edge_index=edge_index)

print(data)
