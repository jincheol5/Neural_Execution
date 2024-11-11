import networkx as nx
import numpy as np
from utils import Data_Generator,Data_Processor

graph=Data_Generator.generate_test_graph_for_check_bellman_ford()


print("Start x: ")
for node_idx in graph.nodes():
    print(graph.nodes[node_idx]['x'][0]) 

print("Start predecessor: ")
for node_idx in graph.nodes():
    print(graph.nodes[node_idx]['predecessor'][0]) 

graph,step_x,step_p=Data_Processor.compute_bellman_ford_step(graph=graph,init=True,source_id=0)

print("Initialize x: ")
for node_idx in graph.nodes():
    print(graph.nodes[node_idx]['x'][0]) 

print("Initialize predecessor: ")
for node_idx in graph.nodes():
    print(graph.nodes[node_idx]['predecessor'][0]) 

graph,step_x,step_p=Data_Processor.compute_bellman_ford_step(graph=graph,source_id=0)

print("Step-1 x: ")
for node_idx in graph.nodes():
    print(graph.nodes[node_idx]['x'][0]) 

print("Step-1 predecessor: ")
for node_idx in graph.nodes():
    print(graph.nodes[node_idx]['predecessor'][0]) 

graph,step_x,step_p=Data_Processor.compute_bellman_ford_step(graph=graph,source_id=0)

print("Step-2 x: ")
for node_idx in graph.nodes():
    print(graph.nodes[node_idx]['x'][0]) 

print("Step-2 predecessor: ")
for node_idx in graph.nodes():
    print(graph.nodes[node_idx]['predecessor'][0]) 