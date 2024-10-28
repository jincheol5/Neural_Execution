import networkx as nx
from utils import Data_Loader
from torch_geometric.utils.convert import from_networkx

dl=Data_Loader()
train_graph_list=dl.load_pickle(file_name="train_graph_list")
data=from_networkx(train_graph_list[0])

print(data)