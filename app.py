import networkx as nx
import numpy as np
from utils import Data_Analysis,Data_Loader

dl=Data_Loader()
test_graph_list=dl.load_pickle(file_name="test_4_community_graph_list")

test_graph=test_graph_list[1]

da=Data_Analysis()
dic=da.find_top_10_sources_with_reachability_networkx_graph(graph=test_graph)

for key, value in dic.items():
    print(f"{key}: {value}")