import networkx as nx
import numpy as np
from utils import Data_Generator,Data_Processor

graph=nx.Graph()
graph.add_nodes_from([0,1,2,3,4])
graph.add_edges_from([(0,1), (0,2), (1,3), (2,3), (3,4)])
graph=Data_Generator.set_self_loop(graph=graph)
# set x feature
for node_idx in graph.nodes():
    graph.nodes[node_idx]['x']=[0.0]
    graph.nodes[node_idx]['p']=[node_idx]
graph[0][0]['w'] = [0.0]
graph[1][1]['w'] = [0.0]
graph[2][2]['w'] = [0.0]
graph[3][3]['w'] = [0.0]
graph[4][4]['w'] = [0.0]
graph[0][1]['w'] = [1.0]
graph[0][2]['w'] = [1.0]
graph[1][3]['w'] = [1.0]
graph[2][3]['w'] = [1.0]
graph[3][4]['w'] = [1.0]


### check bfs
init=True
for i in range(4):
    if init:
        graph_0,x_0=Data_Processor.compute_bfs_step(graph=graph,source_id=0,init=True)
        graph=graph_0
        init=False
        print(x_0)
    else:
        graph_t,x_t=Data_Processor.compute_bfs_step(graph=graph,source_id=0)
        graph=graph_t
        print(x_t)

label=Data_Processor.compute_reachability(graph=graph,source_id=0)
print()
print("label: ")
print(label)

### check b-f
# init=True
# for i in range(4):
#     if init:
#         graph_0,_,x_0=Data_Processor.compute_bellman_ford_step(graph=graph,source_id=0,init=True)
#         graph=graph_0
#         init=False
#         print(x_0)
#     else:
#         graph_t,_,x_t=Data_Processor.compute_bellman_ford_step(graph=graph,source_id=0)
#         graph=graph_t
#         print(x_t)

# _,label=Data_Processor.compute_shortest_path_and_predecessor(graph=graph,source_id=0)
# print()
# print("label: ")
# print(label)