import os
import networkx as nx
import pickle
import random
import copy
import pandas as pd
import numpy as np
import torch


class Data_Generator:
    @staticmethod
    def set_self_loop(graph): 
        self_loof_edge_list = [(node, node) for node in graph.nodes()] 
        graph.add_edges_from(self_loof_edge_list) 
        return graph

    @staticmethod
    def set_edge_weight(graph):
        for source,target in graph.edges(): 
            graph[source][target]['w'] = [random.uniform(0.2, 1)] 
        return graph

    @staticmethod
    def set_graph(graph):
        # set selp loop, edge weight, node feature, predecessor
        graph=Data_Generator.set_self_loop(graph=graph)
        graph=Data_Generator.set_edge_weight(graph=graph)
        for node_idx in graph.nodes():
            graph.nodes[node_idx]['x']=[0.0]
            graph.nodes[node_idx]['p']=[node_idx]
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
    def generate_graph_list_dict(graph_num,node_num,edge_probability=0.5):
        ladder_graph_list=[]
        grid_2d_graph_list=[]
        tree_graph_list=[]
        Erdos_Renyi_graph_list=[]
        Barabasi_Albert_graph_list=[]
        community_graph_list=[]

        # generate
        for _ in range(graph_num):

            # generate ladder graph
            ladder_graph=nx.ladder_graph(n=node_num)
            ladder_graph_list.append(ladder_graph)

            # generate grid 2D graph
            grid_2d_graph=nx.grid_2d_graph(m=node_num,n=node_num) # node id=(x,y), edge=((0,0),(0,1)) 형태를 가짐 
            grid_2d_graph=Data_Generator.convert_grid_2d_to_int_id(grid_2d_graph)
            grid_2d_graph_list.append(grid_2d_graph)

            # generate tree graph
            tree_graph=nx.random_tree(n=node_num)
            tree_graph_list.append(tree_graph)

            # generate Erdos-Renyi graph
            Erdos_Renyi_graph=nx.erdos_renyi_graph(node_num,edge_probability)
            Erdos_Renyi_graph_list.append(Erdos_Renyi_graph)

            # generate Barabasi-Albert graph
            Barabasi_Albert_graph = nx.barabasi_albert_graph(n=node_num, m=2)
            Barabasi_Albert_graph_list.append(Barabasi_Albert_graph)

            # generate 4 community graph
            sub_node_num=node_num//4
            sub_graphs=[nx.erdos_renyi_graph(sub_node_num,p=0.3) for _ in range(4)]
            community_graph=sub_graphs[0]
            for i in range(1, len(sub_graphs)):
                community_graph=nx.disjoint_union(community_graph,sub_graphs[i])

                # 기존 그래프의 노드와 새로 추가된 그래프의 노드 범위 구하기
                G_nodes = list(range(len(community_graph) - len(sub_graphs[i]), len(community_graph) - len(sub_graphs[i]) + len(sub_graphs[i])))
                H_nodes = list(range(len(community_graph) - len(sub_graphs[i]), len(community_graph)))

                # 0.01의 확률로 두 커뮤니티 사이에 에지 생성
                size = len(G_nodes) * len(H_nodes)
                number_of_edges = np.sum(np.random.uniform(size=size) <= 0.01)
                g_nodes_to_connect = np.random.choice(G_nodes, replace=True, size=number_of_edges)
                h_nodes_to_connect = np.random.choice(H_nodes, replace=True, size=number_of_edges)
                edges = list(zip(g_nodes_to_connect, h_nodes_to_connect))

                # 에지 추가
                community_graph.add_edges_from(edges)
            community_graph_list.append(community_graph)

        # set selp loop, edge weight, node feature
        for graph_list in [ladder_graph_list,grid_2d_graph_list,tree_graph_list,Erdos_Renyi_graph_list,Barabasi_Albert_graph_list,community_graph_list]:
            for graph in graph_list:
                graph=Data_Generator.set_graph(graph=graph)
        
        graph_list_dict={}
        graph_list_dict['ladder']=ladder_graph_list
        graph_list_dict['grid']=grid_2d_graph_list
        graph_list_dict['tree']=tree_graph_list
        graph_list_dict['erdos_renyi']=Erdos_Renyi_graph_list
        graph_list_dict['barabasi_albert']=Barabasi_Albert_graph_list
        graph_list_dict['community']=community_graph_list

        return graph_list_dict
    
    @staticmethod
    def generate_4_community_graph_list(graph_num,node_num):
        graph_list=[]

        # generate
        for _ in range(graph_num):
        # generate 4 community graph
            sub_node_num=node_num//4
            sub_graphs=[nx.erdos_renyi_graph(sub_node_num,p=0.3) for _ in range(4)]
            community_graph=sub_graphs[0]
            for i in range(1, len(sub_graphs)):
                community_graph=nx.disjoint_union(community_graph,sub_graphs[i])

                # 기존 그래프의 노드와 새로 추가된 그래프의 노드 범위 구하기
                G_nodes = list(range(len(community_graph) - len(sub_graphs[i]), len(community_graph) - len(sub_graphs[i]) + len(sub_graphs[i])))
                H_nodes = list(range(len(community_graph) - len(sub_graphs[i]), len(community_graph)))

                # 0.01의 확률로 두 커뮤니티 사이에 에지 생성
                size = len(G_nodes) * len(H_nodes)
                number_of_edges = np.sum(np.random.uniform(size=size) <= 0.01)
                g_nodes_to_connect = np.random.choice(G_nodes, replace=True, size=number_of_edges)
                h_nodes_to_connect = np.random.choice(H_nodes, replace=True, size=number_of_edges)
                edges = list(zip(g_nodes_to_connect, h_nodes_to_connect))

                # 에지 추가
                community_graph.add_edges_from(edges)
            graph_list.append(community_graph)

        # set selp loop, edge weight, node feature
        for train_graph in graph_list:
            train_graph=Data_Generator.set_graph(graph=train_graph)
        return graph_list


class Data_Loader:
    pickle_path=os.path.join(os.getcwd(),'data')
    dataset_path=os.path.join('..','data')

    @staticmethod
    def save_pickle(data,file_name):
        file_name=file_name+".pkl"
        with open(os.path.join(Data_Loader.pickle_path,file_name),'wb') as f:
            pickle.dump(data,f)
        print("Save "+file_name)

    @staticmethod
    def load_pickle(file_name):
        file_name=file_name+".pkl"
        with open(os.path.join(Data_Loader.pickle_path,file_name),'rb') as f:
            data=pickle.load(f)
        print("Load "+file_name)
        return data

    @staticmethod
    def load_graph(dataset_name):
        load_path=os.path.join(Data_Loader.dataset_path,dataset_name)
        x_df=pd.read_csv(os.path.join(load_path,"x.csv"))
        edge_index_df=pd.read_csv(os.path.join(load_path,"edge_index.csv"))

        node_num=x_df.shape[0]

        graph=nx.Graph()
        graph.add_nodes_from(range(node_num))
        graph.add_edges_from(zip(edge_index_df['source'], edge_index_df['target']))

        return graph


class Data_Processor:
    
    @staticmethod
    def compute_bfs_step(graph,source_id=0,init=False):
        N=graph.number_of_nodes()
        x_t=torch.zeros((N,1),dtype=torch.float32)
        for idx in range(N):
            x_t[idx][0]=graph.nodes[idx]['x'][0]

        if init:
            for idx in range(N):
                graph.nodes[idx]['x'][0]=0.0
                x_t[idx][0]=0.0
            graph.nodes[source_id]['x'][0]=1.0
            x_t[source_id][0]=1.0

            return graph,x_t
        else:
            graph_t=copy.deepcopy(graph) # 순회 내에서 업데이트 결과가 이후 결과에 영향을 미치지 않도록 복사 -> edge 순서 관계없이 각 단계 결과값 예측 가능
            for idx in range(N):
                if graph.nodes[idx]['x'][0] == 1.0:
                    for neighbor in graph.neighbors(idx):
                        if graph.nodes[neighbor]['x'][0] == 0.0:
                            graph_t.nodes[neighbor]['x'][0] = 1.0
                            x_t[neighbor][0]=1.0

            return graph_t,x_t

    @staticmethod
    def compute_bellman_ford_step(graph,source_id=0,init=False):
        N=graph.number_of_nodes()
        x_t=torch.zeros((N,1),dtype=torch.float32)
        p_t=torch.zeros((N,1),dtype=torch.int)
        for idx in range(N):
            x_t[idx][0]=graph.nodes[idx]['x'][0]
            p_t[idx][0]=graph.nodes[idx]['p'][0]

        if init:
            # compute longest distance
            copy_graph=copy.deepcopy(graph)
            Data_Processor.convert_edge_attr_to_float(copy_graph)
            _, distance_dic=nx.bellman_ford_predecessor_and_distance(G=copy_graph, source=source_id, weight='weight')
            longest_shortest_path_distance=max(distance_dic.values())
            longest_shortest_path_distance+=1

            # initialize 
            for idx in range(N):
                graph.nodes[idx]['x'][0]=longest_shortest_path_distance
                x_t[idx][0]=longest_shortest_path_distance
                graph.nodes[idx]['p'][0]=idx
                p_t[idx][0]=graph.nodes[idx]['p'][0]=idx
            graph.nodes[source_id]['x'][0]=0.0
            x_t[source_id][0]=0.0

            print(x_t)

            return graph,p_t,x_t
        else:
            graph_t=copy.deepcopy(graph) # edge 순서에 영향 받지 않기 위해 복사 -> edge 순서 상관없이 각 단계 결과값 예측 가능
            for src,tar in list(graph.edges()): # edge=(src,tar)
                if graph.nodes[src]['x'][0]+graph.edges[(src, tar)]['w'][0]<graph.nodes[tar]['x'][0]:
                    graph_t.nodes[tar]['x'][0]=graph.nodes[src]['x'][0]+graph.edges[(src, tar)]['w'][0]
                    x_t[tar][0]=graph.nodes[src]['x'][0]+graph.edges[(src, tar)]['w'][0]
                    graph_t.nodes[tar]['p'][0]=src
                    p_t[tar][0]=src

            return graph_t,p_t,x_t

    @staticmethod
    def compute_reachability(graph,source_id):
        nodes=list(graph.nodes())
        result_tensor=torch.zeros((len(nodes),1), dtype=torch.float32) # result_tensor=(N,1)

        for tar in nodes:
            if nx.has_path(graph,source=source_id,target=tar):
                result_tensor[tar][0]=1.0

        return result_tensor
    
    @staticmethod
    def convert_edge_attr_to_float(graph): # single_source_dijkstra_path_length() 함수를 위한 edge_attr 값 처리
        for u, v, data in graph.edges(data=True):
            if isinstance(data['w'], list) and len(data['w']) > 0:
                data['w'] = float(data['w'][0])  # 리스트의 첫 번째 값을 실수형으로 변환하여 'w'에 저장

    @staticmethod
    def compute_shortest_path_and_predecessor(graph,source_id):
        nodes=list(graph.nodes())
        predecessor_tensor=torch.zeros((len(nodes),1), dtype=torch.int) # predecessor_tensor=(N,1)
        distance_tensor=torch.zeros((len(nodes),1), dtype=torch.float32) # distance_tensor=(N,1)

        Data_Processor.convert_edge_attr_to_float(graph)

        predecessor_dic, distance_dic = nx.bellman_ford_predecessor_and_distance(G=graph, source=source_id, weight='w') # 연결되지 않은 노드=key도 없음, source 노드=key는 있지만 value=[]
        predecessor_dic[source_id]=[source_id]

        for tar in nodes:
            if tar in predecessor_dic:
                predecessor_tensor[tar][0]=predecessor_dic[tar][0]
                distance_tensor[tar][0]=distance_dic[tar]
            else:
                predecessor_tensor[tar][0]=tar
        return predecessor_tensor,distance_tensor
