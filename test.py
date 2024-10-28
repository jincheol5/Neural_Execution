import torch
import torch_scatter

# GPU에서 텐서 생성
x = torch.tensor([1.0, 2.0, 3.0], device='cuda')
index = torch.tensor([0, 1, 2], device='cuda')

# scatter 연산 테스트
result = torch_scatter.scatter_add(x, index, dim=0, dim_size=3)
print(result.device)
