import os
import random
import numpy as np
import torch
from utils import Data_Loader,Data_Generator
from model import BFS_Neural_Execution
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

dl=Data_Loader()
dg=Data_Generator()
model=BFS_Neural_Execution(hidden_dim=32)
model_trainer=Model_Trainer(model=model)

train_graph_list=dl.load_pickle(file_name="train_graph_list")
test_graph=dg.generate_test_graph()

model_trainer.train_bfs(train_graph_list=train_graph_list,hidden_dim=32)
model_trainer.test_bfs_one(test_graph=test_graph,hidden_dim=32)