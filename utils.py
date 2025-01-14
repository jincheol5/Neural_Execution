import os
import random
import copy
import pickle
import networkx as nx
import pandas as pd
import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss
import torch.nn.functional as F
from torch_geometric.utils.convert import from_networkx

class Data_Generator:
    @staticmethod
    def set_self_loop(graph): 
        self_loof_edge_list=[(node, node) for node in graph.nodes()] 
        graph.add_edges_from(self_loof_edge_list)

    @staticmethod
    def set_edge_feature(graph):
        for edge in graph.edges():
            graph.edges[edge]['w']=random.uniform(0.2,1.0)

    @staticmethod
    def set_graph(graph):
        Data_Generator.set_self_loop(graph=graph)
        Data_Generator.set_edge_feature(graph=graph)

    @staticmethod
    def generate_graph_list_dict(graph_num,node_num,edge_probability=0.1):
        ladder_graph_list=[]
        tree_graph_list=[]
        Erdos_Renyi_graph_list=[]
        Barabasi_Albert_graph_list=[]
        community_graph_list=[]

        # generate
        for _ in range(graph_num):
            # generate ladder graph
            ladder_graph=nx.ladder_graph(n=int(node_num/2))
            directed_ladder_graph=nx.DiGraph()
            directed_ladder_graph.add_nodes_from(ladder_graph.nodes())
            for u,v in ladder_graph.edges():
                if np.random.rand()<0.5:
                    directed_ladder_graph.add_edge(u,v)
                else:
                    directed_ladder_graph.add_edge(v,u)
            ladder_graph_list.append(directed_ladder_graph)

            # generate tree graph
            tree_graph=nx.random_tree(n=node_num)
            directed_tree_graph=nx.DiGraph()
            directed_tree_graph.add_nodes_from(tree_graph.nodes())
            for parent,child in nx.bfs_edges(tree_graph,source=0):
                directed_tree_graph.add_edge(parent,child)
            tree_graph_list.append(directed_tree_graph)

            # generate Erdos-Renyi graph
            Erdos_Renyi_graph=nx.erdos_renyi_graph(node_num,edge_probability,directed=True)
            Erdos_Renyi_graph_list.append(Erdos_Renyi_graph)

            # generate Barabasi-Albert graph
            Barabasi_Albert_graph=nx.barabasi_albert_graph(n=node_num, m=2)
            directed_Barabasi_Albert_graph=nx.DiGraph()
            directed_Barabasi_Albert_graph.add_nodes_from(Barabasi_Albert_graph.nodes())
            for u,v in Barabasi_Albert_graph.edges():
                if np.random.rand()<0.5:
                    directed_Barabasi_Albert_graph.add_edge(u,v)
                else:
                    directed_Barabasi_Albert_graph.add_edge(v,u)
            Barabasi_Albert_graph_list.append(directed_Barabasi_Albert_graph)

            # generate 4 community graph
            sub_node_num=node_num//4
            sub_graphs=[nx.erdos_renyi_graph(sub_node_num,p=0.5,directed=True) for _ in range(4)]
            community_graph=sub_graphs[0]
            for i in range(1, len(sub_graphs)):
                community_graph=nx.disjoint_union(community_graph,sub_graphs[i])

                # 기존 그래프의 노드와 새로 추가된 그래프의 노드 범위 구하기
                G_nodes=list(range(len(community_graph)-len(sub_graphs[i]),len(community_graph)-len(sub_graphs[i])+len(sub_graphs[i])))
                H_nodes=list(range(len(community_graph)-len(sub_graphs[i]),len(community_graph)))

                # 0.01의 확률로 두 커뮤니티 사이에 에지 생성
                size=len(G_nodes)*len(H_nodes)
                number_of_edges=np.sum(np.random.uniform(size=size)<=0.01)
                g_nodes_to_connect=np.random.choice(G_nodes,replace=True, size=number_of_edges)
                h_nodes_to_connect=np.random.choice(H_nodes,replace=True, size=number_of_edges)
                edges=[(g,h) if np.random.rand()<0.5 else (h,g) for g,h in zip(g_nodes_to_connect,h_nodes_to_connect)] # 방향성 고려

                # 에지 추가
                community_graph.add_edges_from(edges)
            community_graph_list.append(community_graph)

        # set selp loop, edge weight, edge time list
        for graph_list in [ladder_graph_list,tree_graph_list,Erdos_Renyi_graph_list,Barabasi_Albert_graph_list,community_graph_list]:
            for graph in graph_list:
                Data_Generator.set_graph(graph=graph)

        graph_list_dict={}
        graph_list_dict['ladder']=ladder_graph_list
        graph_list_dict['tree']=tree_graph_list
        graph_list_dict['erdos_renyi']=Erdos_Renyi_graph_list
        graph_list_dict['barabasi_albert']=Barabasi_Albert_graph_list
        graph_list_dict['community']=community_graph_list

        return graph_list_dict

class Data_Loader:
    train_dataset_path=os.path.join('..','data','ngae')
    dataset_path=os.path.join('..','data')

    @staticmethod
    def save_to_pickle(data,file_name):
        file_name=file_name+".pkl"
        with open(os.path.join(Data_Loader.train_dataset_path,file_name),'wb') as f:
            pickle.dump(data,f)
        print(f"Save {file_name}")

    @staticmethod
    def load_from_pickle(file_name):
        file_name=file_name+".pkl"
        with open(os.path.join(Data_Loader.train_dataset_path,file_name),'rb') as f:
            data=pickle.load(f)
        print(f"Load {file_name}")
        return data

    @staticmethod
    def save_model_parameter(model,model_name="model_parameter"):
        file_name=model_name+".pt"
        save_path=os.path.join(os.getcwd(),"inference",file_name)
        torch.save(model.state_dict(),save_path)
        print(f"Save model parameter: {model_name}")

class Graph_Algorithm:
    @staticmethod
    def compute_bfs_step(graph,source_id=0,init=False,Q=None):
        N=graph.number_of_nodes()
        Q_next=[]
        if init:
            graph.nodes[source_id]['r']=1.0
            Q_next.append(source_id)
        else:
            for node in Q:
                for _,neighbor in graph.out_edges(node): # neighbor=(src,tar)
                    if graph.nodes[neighbor]['r']==0.0:
                        graph.nodes[neighbor]['r']=1.0
                        Q_next.append(neighbor)
        return Q_next

    @staticmethod
    def compute_reachability(graph,source_id=0):
        Q=Graph_Algorithm.compute_bfs_step(graph=graph,source_id=source_id,init=True)
        while Q:
            Q_next=Graph_Algorithm.compute_bfs_step(graph=graph,source_id=source_id,init=False,Q=Q)
            Q=Q_next
        data=from_networkx(G=graph,group_node_attrs=['r'])
        r=data.x # [N,1]
        return r 

    @staticmethod
    def compute_reachability_has_path(graph,source_id):
        nodes=list(graph.nodes())
        result_tensor=torch.zeros((len(nodes),1),dtype=torch.float32) # result_tensor=(N,1)
        for tar in nodes:
            if nx.has_path(graph,source=source_id,target=tar):
                result_tensor[tar][0]=1.0
        return result_tensor

class Metrics:
    @staticmethod
    def compute_BFS_from_logit(logit):
        return F.sigmoid(logit)

    @staticmethod
    def compute_tau_from_logit(logit):
        return F.sigmoid(logit)

    @staticmethod
    def compute_BFS_accuracy(predict,label):
        """
        각 source 마다 수행
        predict: [N,1], logit
        label: [N,1]
        """
        N=predict.size(0)
        predict_class=(predict>=0.5).float()
        correct=(predict_class==label).sum().item()
        acc=correct/N
        return acc

    @staticmethod
    def compute_tau_accuracy(predict,label):
        """
        모든 source 에 대해 한번 수행
        tau: 1=continue, 0=terminate
        tau_predict: [N,1], logit
        tau_label: [N,1]
        """
        N=predict.size(0)
        tau_predict=(predict>=0.5).float()
        correct=(tau_predict==label).sum().item()
        acc=correct/N
        return acc

    @staticmethod
    def compute_tau_label(Q):
        if not Q:
            return torch.tensor([[0.0]])
        else:
            return torch.tensor([[1.0]])
    
    @staticmethod
    def compute_BFS_loss(y,y_label):
        """
        criterion: BCEWithLogitsLoss()
        y: [N,1], logit
        y_label: [N,1]
        """
        criterion=BCEWithLogitsLoss()
        loss=criterion(y,y_label)
        return loss

    @staticmethod
    def compute_tau_loss(tau,tau_label):
        """
        criterion: BCEWithLogitsLoss()
        tau: [1,1], logit
        tau_label: [1,1]
        """
        criterion=BCEWithLogitsLoss()
        loss=criterion(tau,tau_label)
        return loss