import networkx as nx
import numpy as np
from utils import Data_Analysis

da=Data_Analysis()
dic=da.find_top_10_sources_with_reachability(dataset_name="Cora")

for key, value in dic.items():
    print(f"{key}: {value}")