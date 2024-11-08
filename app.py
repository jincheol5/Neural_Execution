import networkx as nx
import numpy as np
from utils import Data_Analysis,Data_Loader,Data_Generator

graph=Data_Loader.load_graph(dataset_name="Cora")
graph=Data_Generator.set_graph(graph=graph)
top_10_src_dic=Data_Analysis.get_reachability_ratio(graph=graph)

print(top_10_src_dic)