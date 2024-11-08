import os
import random
import numpy as np
import torch
from utils import Data_Loader,Data_Analysis
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

### Evaluation about Cora dataset
graph=Data_Loader.load_graph(dataset_name="Cora")
top_10_src_dic=Data_Analysis.get_reachability_ratio(graph=graph)
top_10_src_list=sorted(top_10_src_dic.keys())
ml=Model_Trainer()
ml.evaluate_bfs_dataset(test_graph=graph,model_file_name="neural_execution_bfs",src_list=top_10_src_list,hidden_dim=32)
