import os
import random
import numpy as np
import torch
from utils import Data_Loader
from model import BFS_Neural_Execution,BF_Neural_Execution
from model_train import Model_Trainer

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

### Load data
train_graph_list_dict=Data_Loader.load_pickle(file_name="train_graph_list_dict")
test_graph_list_dict_20=Data_Loader.load_pickle(file_name="test_graph_list_dict_20")
test_graph_list_dict_30=Data_Loader.load_pickle(file_name="test_graph_list_dict_30")
test_graph_list_dict_50=Data_Loader.load_pickle(file_name="test_graph_list_dict_50")

### BFS
# model_file_name="neural_execution_bfs"
# Model_Trainer.evaluate_bfs(test_graph_list_dict=test_graph_list_dict_20,model_file_name=model_file_name,hidden_dim=32)
# print("Finish to evaluate test_graph_list_dict_20.")
# print()
# Model_Trainer.evaluate_bfs(test_graph_list_dict=test_graph_list_dict_30,model_file_name=model_file_name,hidden_dim=32)
# print("Finish to evaluate test_graph_list_dict_30.")
# print()
# Model_Trainer.evaluate_bfs(test_graph_list_dict=test_graph_list_dict_50,model_file_name=model_file_name,hidden_dim=32)
# print("Finish to evaluate test_graph_list_dict_50.")
# print()


### Bellman-Ford
model_file_name="neural_execution_bf_distance"
Model_Trainer.evaluate_bf_distance(test_graph_list_dict=test_graph_list_dict_20,model_file_name=model_file_name,hidden_dim=32)
print("Finish to evaluate test_graph_list_dict_20.")
print()
Model_Trainer.evaluate_bf_distance(test_graph_list_dict=test_graph_list_dict_30,model_file_name=model_file_name,hidden_dim=32)
print("Finish to evaluate test_graph_list_dict_30.")
print()
Model_Trainer.evaluate_bf_distance(test_graph_list_dict=test_graph_list_dict_50,model_file_name=model_file_name,hidden_dim=32)
print("Finish to evaluate test_graph_list_dict_50.")
print()