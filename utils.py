import os
import networkx as nx
import pickle
import random
import copy

class DataGenerator:
    def __init__(self):
        pass

    def set_self_loop(self,graph): 
        self_loof_edge_list = [(node, node) for node in graph.nodes()] 
        graph.add_edges_from(self_loof_edge_list) 
        return graph

    def set_edge_weight(self,graph):
        for source,target in graph.edges(): 
            graph[source][target]['weight'] = random.uniform(0.2, 1) 
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
                graph.nodes[node_idx]['bfs']=0.0
            graph_list.append(graph)

        return graph_list

class DataLoader:
    def __init__(self):
        self.file_path=os.path.join(os.getcwd(),'data')

    def save_data(self,data,file_name):
        file_name=file_name+".pkl"
        with open(os.path.join(self.file_path,file_name),'wb') as f:
            pickle.dump(data,f)
        print("Save "+file_name)

    def load_data(self,file_name):
        file_name=file_name+".pkl"
        with open(os.path.join(self.file_path,file_name),'rb') as f:
            data=pickle.load(f)
        print("Load "+file_name)
        return data

class DataProcessor:
    def __init__(self):
        pass
    
    def compute_bfs_step(self,graph,init=False,source_index=0):
        copy_graph=copy.deepcopy(graph)

        if init:
            copy_graph.nodes[source_index]['bfs']=1.0
            return copy_graph

        for node_idx in graph.nodes():
            if graph.nodes[node_idx]['bfs'] == 1.0:
                for neighbor in graph.neighbors(node_idx):
                    if graph.nodes[neighbor]['bfs'] == 0.0:
                        copy_graph.nodes[neighbor]['bfs'] = 1.0

        return copy_graph
