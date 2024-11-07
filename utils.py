import os
import networkx as nx
import pickle
import random
import copy
import numpy as np
import torch
from torch_geometric.data import Data


class Data_Generator:

    @staticmethod
    def set_self_loop(graph): 
        self_loof_edge_list = [(node, node) for node in graph.nodes()] 
        graph.add_edges_from(self_loof_edge_list) 
        return graph

    @staticmethod
    def set_edge_weight(graph):
        for source,target in graph.edges(): 
            graph[source][target]['edge_attr'] = [random.uniform(0.2, 1)] 
        return graph

    @staticmethod
    def convert_grid_2d_to_int_id(graph):
        # 그래프의 행과 열 크기를 가져옵니다.
        m = max(x for x, y in graph.nodes()) + 1
        n = max(y for x, y in graph.nodes()) + 1

        # 노드 좌표를 정수형 ID로 매핑하는 딕셔너리 생성
        node_mapping = {(i, j): i * n + j for i in range(m) for j in range(n)}

        # 새로운 그래프에 정수형 노드 ID와 엣지를 추가합니다.
        new_graph = nx.Graph()
        for (node1, node2) in graph.edges():
            # 좌표형 ID를 정수형 ID로 변환하여 엣지를 추가
            new_graph.add_edge(node_mapping[node1], node_mapping[node2])

        return new_graph

    @staticmethod
    def generate_graph_list(graph_num,node_num,edge_probability=0.5):
        graph_list=[]

        # generate
        for _ in range(graph_num):

            # generate ladder graph
            ladder_graph=nx.ladder_graph(n=node_num)
            graph_list.append(ladder_graph)

            # generate grid 2D graph
            grid_2d_graph=nx.grid_2d_graph(m=node_num,n=node_num) # node id = (x,y), edge = ((0,0),(0,1)) 형태를 가짐 
            grid_2d_graph=Data_Generator.convert_grid_2d_to_int_id(grid_2d_graph)
            graph_list.append(grid_2d_graph)

            # generate tree graph
            tree_graph=nx.random_tree(n=node_num)
            graph_list.append(tree_graph)

            # generate Erdos-Renyi graph
            Erdos_Renyi_graph=nx.erdos_renyi_graph(node_num,edge_probability)
            graph_list.append(Erdos_Renyi_graph)

            # generate Barabasi-Albert graph
            Barabasi_Albert_graph = nx.barabasi_albert_graph(n=node_num, m=2)
            graph_list.append(Barabasi_Albert_graph)

            # generate 4 community graph
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

        # set selp loop, edge weight, node feature
        for train_graph in graph_list:
            train_graph=Data_Generator.set_self_loop(graph=train_graph)
            train_graph=Data_Generator.set_edge_weight(graph=train_graph)
            for node_idx in train_graph.nodes():
                train_graph.nodes[node_idx]['x']=[0.0]


        return graph_list
    
    @staticmethod
    def generate_test_graph(self):
        graph=nx.Graph() # undirected graph 
        graph.add_edges_from([(0,2), (0,3),(2,1),(3,4)])
        graph=self.set_self_loop(graph=graph)
        graph=self.set_edge_weight(graph=graph)
        for node_idx in graph.nodes():
                graph.nodes[node_idx]['x']=[0.0]
        return graph


class Data_Loader:
    def __init__(self):
        self.pickle_path=os.path.join(os.getcwd(),'data')
        self.dataset_path=os.path.join("..","data")

    def save_pickle(self,data,file_name):
        file_name=file_name+".pkl"
        with open(os.path.join(self.pickle_path,file_name),'wb') as f:
            pickle.dump(data,f)
        print("Save "+file_name)

    def load_pickle(self,file_name):
        file_name=file_name+".pkl"
        with open(os.path.join(self.pickle_path,file_name),'rb') as f:
            data=pickle.load(f)
        print("Load "+file_name)
        return data

class Data_Processor:
    
    @staticmethod
    def compute_bfs_step(graph,init=False,source_id=0):
        copy_graph=copy.deepcopy(graph)
        step_x_label=torch.zeros((len(graph.nodes()),1),dtype=torch.float32) # (num_nodes,1)
        for node_idx in graph.nodes():
            step_x_label[node_idx][0]=graph.nodes[node_idx]['x'][0]

        if init:
            copy_graph.nodes[source_id]['x'][0]=1.0
            step_x_label[source_id][0]=1.0
            return copy_graph, step_x_label

        for node_idx in graph.nodes():
            if graph.nodes[node_idx]['x'][0] == 1.0:
                for neighbor in graph.neighbors(node_idx):
                    if graph.nodes[neighbor]['x'][0] == 0.0:
                        copy_graph.nodes[neighbor]['x'][0] = 1.0
                        step_x_label[neighbor][0]=1.0

        return copy_graph, step_x_label

    @staticmethod
    def compute_reachability(graph,source_id):
        nodes=list(graph.nodes())
        result_tensor=torch.zeros((len(nodes),1), dtype=torch.float32) # (num_nodes,1)

        for tar in nodes:
            if nx.has_path(graph,source=source_id,target=tar):
                result_tensor[tar][0]=1.0

        return result_tensor
