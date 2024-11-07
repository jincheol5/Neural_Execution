import os
import random
import numpy as np
import torch
from utils import Data_Loader,Data_Processor

### seed setting
seed= 42
random.seed(seed) # Python의 기본 random seed 설정
np.random.seed(seed) # NumPy의 random seed 설정
torch.manual_seed(seed) # PyTorch의 random seed 설정
os.environ["PYTHONHASHSEED"] = str(seed)
# CUDA 사용 시, 모든 GPU에서 동일한 seed 사용
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# PyTorch의 난수 생성 결정론적 동작 보장 (동일 연산 결과 보장)
torch.backends.cudnn.deterministic = True 
torch.backends.cudnn.benchmark = False

dl=Data_Loader()
test_graph_list=dl.load_pickle(file_name="test_graph_list")

count=0
for graph in test_graph_list:
    all_reachable=False
    source_id=random.randint(0, graph.number_of_nodes() - 1)
    reach_tensor=Data_Processor.compute_reachability(graph=graph,source_id=source_id)
    all_reachable=torch.all(reach_tensor==1.0)
    if all_reachable:
        count+=1

print("all reachable count: ",count)