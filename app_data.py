import os
import random
import numpy as np
import torch
from utils import Data_Generator,Data_Loader

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

train_graph_list=Data_Generator.generate_graph_list(graph_num=100,node_num=20,edge_probability=0.5)
test_graph_list=[]
for node_num in [20,30,50]:
    test_graph_list=test_graph_list+Data_Generator.generate_graph_list(graph_num=10,node_num=node_num,edge_probability=0.5)
Data_Loader.save_pickle(data=train_graph_list,file_name="train_graph_list")
Data_Loader.save_pickle(data=train_graph_list,file_name="test_graph_list")