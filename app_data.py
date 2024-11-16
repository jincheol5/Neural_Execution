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

train_graph_list_dict=Data_Generator.generate_graph_list_dict(graph_num=100,node_num=20,edge_probability=0.5)
val_graph_list_dict=Data_Generator.generate_graph_list_dict(graph_num=3,node_num=20,edge_probability=0.5)
test_graph_list_dict_20=Data_Generator.generate_graph_list_dict(graph_num=5,node_num=20,edge_probability=0.5)
test_graph_list_dict_30=Data_Generator.generate_graph_list_dict(graph_num=5,node_num=30,edge_probability=0.5)
test_graph_list_dict_50=Data_Generator.generate_graph_list_dict(graph_num=5,node_num=50,edge_probability=0.5)

Data_Loader.save_pickle(data=train_graph_list_dict,file_name="train_graph_list_dict")
Data_Loader.save_pickle(data=val_graph_list_dict,file_name="val_graph_list_dict")
Data_Loader.save_pickle(data=test_graph_list_dict_20,file_name="test_graph_list_dict_20")
Data_Loader.save_pickle(data=test_graph_list_dict_30,file_name="test_graph_list_dict_30")
Data_Loader.save_pickle(data=test_graph_list_dict_50,file_name="test_graph_list_dict_50")