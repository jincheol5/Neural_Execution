import os
import networkx as nx
import pickle
import random
import copy
import torch
from torch_geometric.data import Data


class Data_Generator:
    def __init__(self):
        pass

    def set_self_loop(self,graph): 
        self_loof_edge_list = [(node, node) for node in graph.nodes()] 
        graph.add_edges_from(self_loof_edge_list) 
        return graph

    def set_edge_weight(self,graph):
        for source,target in graph.edges(): 
            graph[source][target]['edge_attr'] = [random.uniform(0.2, 1)] 
        return graph

    def generate_graph_list(self,num_graph,num_node):
        graph_list=[]

        for _ in range(num_graph):
            # generate Erdos Renyi graph
            edge_probability=0.5
            graph=nx.erdos_renyi_graph(num_node,edge_probability)
            graph=self.set_self_loop(graph=graph)
            graph=self.set_edge_weight(graph=graph)
            for node_idx in graph.nodes():
                graph.nodes[node_idx]['x']=[0.0]
            graph_list.append(graph)

        return graph_list
    
    def generate_test_graph(self):
        graph=nx.Graph() # undirected graph 
        graph.add_edges_from([(0,2), (0,3),(2,1),(3,4)])
        graph=self.set_self_loop(graph=graph)
        graph=self.set_edge_weight(graph=graph)
        
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
    def __init__(self):
        pass
    
    def compute_bfs_step(self,graph,init=False,source_id=0):
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

    def compute_reachability(self,graph,source_id):
        nodes=list(graph.nodes())
        result_tensor=torch.zeros((len(nodes),1), dtype=torch.float32) # (num_nodes,1)

        for tar in nodes:
            if nx.has_path(graph,source=source_id,target=tar):
                result_tensor[tar][0]=1.0

        return result_tensor
