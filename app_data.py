import os
import random
import numpy as np
import torch
from utils import Data_Generator,Data_Loader

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
Save train/val graph (node: 20, 50 ,100)
"""
train_graph_20_list_dict=Data_Generator.generate_graph_list_dict(graph_num=100,node_num=20,edge_probability=0.5)
train_graph_50_list_dict=Data_Generator.generate_graph_list_dict(graph_num=100,node_num=20,edge_probability=0.5)
train_graph_100_list_dict=Data_Generator.generate_graph_list_dict(graph_num=100,node_num=20,edge_probability=0.5)
val_graph_20_list_dict=Data_Generator.generate_graph_list_dict(graph_num=3,node_num=20,edge_probability=0.1)
val_graph_50_list_dict=Data_Generator.generate_graph_list_dict(graph_num=3,node_num=20,edge_probability=0.1)
val_graph_100_list_dict=Data_Generator.generate_graph_list_dict(graph_num=3,node_num=20,edge_probability=0.1)

Data_Loader.save_to_pickle(data=train_graph_20_list_dict,file_name="train_graph_20_list_dict")
Data_Loader.save_to_pickle(data=train_graph_50_list_dict,file_name="train_graph_50_list_dict")
Data_Loader.save_to_pickle(data=train_graph_100_list_dict,file_name="train_graph_100_list_dict")
Data_Loader.save_to_pickle(data=val_graph_20_list_dict,file_name="val_graph_20_list_dict")
Data_Loader.save_to_pickle(data=val_graph_50_list_dict,file_name="val_graph_50_list_dict")
Data_Loader.save_to_pickle(data=val_graph_100_list_dict,file_name="val_graph_100_list_dict")

"""
Save test graph (node: 50, 100, 1000)
"""
test_graph_50_list_dict=Data_Generator.generate_graph_list_dict(graph_num=5,node_num=50,edge_probability=0.1)
test_graph_100_list_dict=Data_Generator.generate_graph_list_dict(graph_num=5,node_num=100,edge_probability=0.1)
test_graph_1000_list_dict=Data_Generator.generate_graph_list_dict(graph_num=5,node_num=1000,edge_probability=0.1)

Data_Loader.save_to_pickle(data=test_graph_50_list_dict,file_name="test_graph_50_list_dict")
Data_Loader.save_to_pickle(data=test_graph_100_list_dict,file_name="test_graph_100_list_dict")
Data_Loader.save_to_pickle(data=test_graph_1000_list_dict,file_name="test_graph_1000_list_dict")