import networkx as nx
from utils import Data_Generator,Graph_Algorithm
from model_train import Execution_Engine
import torch
from torch_geometric.utils.convert import from_networkx

"""
Create simple graph
"""
graph=nx.DiGraph()
graph.add_nodes_from([0,1,2,3,4,5])
graph.add_edge(0,1)
graph.add_edge(0,2)
graph.add_edge(0,3)
graph.add_edge(0,4)
graph.add_edge(2,4)
graph.add_edge(3,4)
graph.add_edge(4,1)
graph.add_edge(4,5)
Data_Generator.set_self_loop(graph=graph)
Data_Generator.set_edge_feature(graph=graph)

"""
Test BFS step result
"""
Execution_Engine.initialize(graph=graph)
Q=Graph_Algorithm.compute_bfs_step(graph=graph,source_id=0,init=True)
data=from_networkx(G=graph,group_node_attrs=['r'])
r=data.x # [N,1]
print(r)
while Q:
    Q_next=Graph_Algorithm.compute_bfs_step(graph=graph,source_id=0,init=False,Q=Q)
    data=from_networkx(G=graph,group_node_attrs=['r'])
    r=data.x # [N,1]
    print(r)
    Q=Q_next

"""
Test BFS last result
"""
Execution_Engine.initialize(graph=graph)
custom_bfs_result=Graph_Algorithm.compute_reachability(graph=graph,source_id=0)
networkx_result=Graph_Algorithm.compute_reachability_has_path(graph=graph,source_id=0)
print(torch.equal(custom_bfs_result,networkx_result))


