import networkx as nx
import numpy as np
from utils import Data_Processor

node_num=16
sub_node_num=node_num//4

sub_graphs=[nx.erdos_renyi_graph(sub_node_num,p=0.3) for _ in range(4)]


graph=sub_graphs[0]
for i in range(1, len(sub_graphs)):
    graph=nx.disjoint_union(graph,sub_graphs[i])

    # 기존 그래프의 노드와 새로 추가된 그래프의 노드 범위 구하기
    G_nodes = list(range(len(graph) - len(sub_graphs[i]), len(graph) - len(sub_graphs[i]) + len(sub_graphs[i])))
    H_nodes = list(range(len(graph) - len(sub_graphs[i]), len(graph)))

    # 0.01의 확률로 두 커뮤니티 사이에 에지 생성
    size = len(G_nodes) * len(H_nodes)
    number_of_edges = np.sum(np.random.uniform(size=size) <= 0.01)
    g_nodes_to_connect = np.random.choice(G_nodes, replace=True, size=number_of_edges)
    h_nodes_to_connect = np.random.choice(H_nodes, replace=True, size=number_of_edges)
    edges = list(zip(g_nodes_to_connect, h_nodes_to_connect))

    # 에지 추가
    graph.add_edges_from(edges)


print(list(graph.nodes))

print(Data_Processor.compute_reachability(graph=graph,source_id=0))