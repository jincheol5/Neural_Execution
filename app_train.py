import os
import random
import numpy as np
import torch
from utils import Data_Loader
from model import NGAE_MPNN_BFS
from model_train import Model_Trainer

### seed setting
seed=42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed) 
os.environ["PYTHONHASHSEED"]=str(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic=True 
torch.backends.cudnn.benchmark=False

"""
Train with 20 node
"""
train_graph_list_dict=Data_Loader.load_from_pickle(file_name="train_graph_20_list_dict")
val_graph_list_dict=Data_Loader.load_from_pickle(file_name="val_graph_20_list_dict")

model=NGAE_MPNN_BFS(x_dim=1,e_dim=1,latent_dim=32,aggr='max')
trained_model=Model_Trainer.train(model=model,train_graph_list_dict=train_graph_list_dict,val_graph_list_dict=val_graph_list_dict,latent_dim=32,epochs=30)
Data_Loader.save_model_parameter(model=trained_model,model_name="NGAE_MPNN_BFS_MAX_20")
