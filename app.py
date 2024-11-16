import networkx as nx
import numpy as np
from utils import Data_Generator,Data_Processor

graph=Data_Generator.generate_test_graph_for_check_bellman_ford()
N=graph.number_of_nodes()
source_id=0

graph_0,_,x_0=Data_Processor.compute_bellman_ford_step(graph=graph,init=True,source_id=source_id)
for node, data in graph_0.nodes(data=True):
    print(f"Node {node}: x = {data.get('x')}")

graph=graph_0

graph_1,_,x_1=Data_Processor.compute_bellman_ford_step(graph=graph,source_id=source_id)
for node, data in graph_1.nodes(data=True):
    print(f"Node {node}: x = {data.get('x')}")


# for u,v,data in graph.edges(data=True):
#     print(f"Edge ({u}, {v}): edge_attr = {data.get('edge_attr')}")

# graph_0,_,x_0=Data_Processor.compute_bellman_ford_step(graph=graph,init=True,source_id=source_id)
# for t in range(N):
#     graph_t,_,x_t=Data_Processor.compute_bellman_ford_step(graph=graph,source_id=source_id)
#     print(x_t)
#     graph=graph_t