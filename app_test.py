import os
import random
import numpy as np
import torch
from utils import Data_Loader
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
test all source
    node: 50
"""
# test_graph_list_dict=Data_Loader.load_from_pickle(file_name="test_graph_50_list_dict")
# Model_Trainer.evaluate(model_file_name="NGAE_MPNN_MAX_BFS_20",graph_list_dict=test_graph_list_dict,model_type="mpnn_max",latent_dim=32)

"""
test all source
    node: 100
"""
# test_graph_list_dict=Data_Loader.load_from_pickle(file_name="test_graph_100_list_dict")
# Model_Trainer.evaluate(model_file_name="NGAE_MPNN_MAX_BFS_20",graph_list_dict=test_graph_list_dict,model_type="mpnn_max",latent_dim=32)

"""
test all source
    node: 1000
"""
test_graph_list_dict=Data_Loader.load_from_pickle(file_name="test_graph_1000_list_dict")
Model_Trainer.evaluate(model_file_name="NGAE_MPNN_MAX_BFS_20",graph_list_dict=test_graph_list_dict,model_type="mpnn_max",latent_dim=32)