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

# class instance 생성
dl=Data_Loader()

### train, test graph list 생성 및 저장
# train = 각 노드 수가 20개인 six-type graph 각 100개씩 => 총 600개
# test = 각 노드 수가 20, 30, 50개인 graph 각 10개씩 => 총 180개 
# train_graph_list=Data_Generator.generate_graph_list(graph_num=100,node_num=20)
test_graph_list=[]
# test_graph_list.extend(Data_Generator.generate_graph_list(graph_num=10,node_num=20))
# test_graph_list.extend(Data_Generator.generate_graph_list(graph_num=10,node_num=30))
# test_graph_list.extend(Data_Generator.generate_graph_list(graph_num=10,node_num=50))
test_graph_list.extend(Data_Generator.generate_4_community_graph_list(graph_num=10,node_num=20))
test_graph_list.extend(Data_Generator.generate_4_community_graph_list(graph_num=10,node_num=30))
test_graph_list.extend(Data_Generator.generate_4_community_graph_list(graph_num=10,node_num=50))

# dl.save_pickle(data=train_graph_list,file_name="train_graph_list")
# dl.save_pickle(data=test_graph_list,file_name="test_graph_list")
dl.save_pickle(data=test_graph_list,file_name="test_4_community_graph_list")


