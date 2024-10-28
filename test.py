import torch
import torch_scatter

# GPU에서 텐서 생성
x = torch.tensor([1, 2, 3], dtype=torch.float32, device='cuda')
index = torch.tensor([0, 1, 2], device='cuda')

# GPU에서 scatter 연산 테스트
try:
    result = torch_scatter.scatter_add(x, index, dim=0, dim_size=3)
    print("CUDA 지원이 가능합니다!")
    print(result)
except RuntimeError as e:
    print("CUDA가 지원되지 않습니다:", e)
