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
            graph[source][target]['edge_attr'] = [random.uniform(0.2, 1)] 
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
    def generate_graph_list(graph_num,node_num,edge_probability=0.5):
        graph_list=[]

        # generate
        for _ in range(graph_num):

            # generate ladder graph
            ladder_graph=nx.ladder_graph(n=node_num)
            graph_list.append(ladder_graph)

            # generate grid 2D graph
            grid_2d_graph=nx.grid_2d_graph(m=node_num,n=node_num) # node id=(x,y), edge=((0,0),(0,1)) 형태를 가짐 
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

    @staticmethod
    def generate_test_graph_for_check_bfs():
        graph=nx.Graph() # undirected graph
        graph.add_nodes_from([0,1,2,3,4]) 
        graph.add_edges_from([(0,2), (0,3)])
        graph=Data_Generator.set_graph(graph=graph)
        return graph
    
    @staticmethod
    def generate_test_graph_for_check_bellman_ford():
        graph=nx.Graph() # undirected graph
        graph.add_nodes_from([0,1,2,3,4,5]) 
        graph.add_edges_from([(0,1), (0,2), (1,2), (1,3), (2,4), (3,5), (4,1), (4,3), (4,5)])
        graph=Data_Generator.set_self_loop(graph=graph)
        
        # set edge weight
        graph[0][1]['edge_attr'] = [0.6]
        graph[0][2]['edge_attr'] = [0.2]
        graph[1][2]['edge_attr'] = [0.2]
        graph[1][3]['edge_attr'] = [0.2]
        graph[2][4]['edge_attr'] = [0.1]
        graph[3][5]['edge_attr'] = [0.2]
        graph[4][1]['edge_attr'] = [0.1]  
        graph[4][3]['edge_attr'] = [0.3]
        graph[4][5]['edge_attr'] = [0.4]

        graph[0][0]['edge_attr'] = [0.0]
        graph[1][1]['edge_attr'] = [0.0]
        graph[2][2]['edge_attr'] = [0.0]
        graph[3][3]['edge_attr'] = [0.0]
        graph[4][4]['edge_attr'] = [0.0]
        graph[5][5]['edge_attr'] = [0.0]

        # set x feature
        for node_idx in graph.nodes():
            graph.nodes[node_idx]['x']=[0.0]
            graph.nodes[node_idx]['p']=[node_idx]

        return graph


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
    def compute_bfs_step(graph,init=False,source_id=0):
        copy_graph=copy.deepcopy(graph)
        step_x_label=torch.zeros((len(graph.nodes()),1),dtype=torch.float32) # step_x_label=(N,1)
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
        result_tensor=torch.zeros((len(nodes),1), dtype=torch.float32) # result_tensor=(N,1)

        for tar in nodes:
            if nx.has_path(graph,source=source_id,target=tar):
                result_tensor[tar][0]=1.0

        return result_tensor
    
    @staticmethod
    def convert_edge_attr_to_float(graph): # single_source_dijkstra_path_length() 함수를 위한 edge_attr 값 처리
        for u, v, data in graph.edges(data=True):
            if isinstance(data['edge_attr'], list) and len(data['edge_attr']) > 0:
                data['weight'] = float(data['edge_attr'][0])  # 리스트의 첫 번째 값을 실수형으로 변환하여 'weight'에 저장

    @staticmethod
    def compute_shortest_path_and_predecessor(graph,source_id):
        nodes=list(graph.nodes())
        predecessor_tensor=torch.zeros((len(nodes),1), dtype=torch.float32) # predecessor_tensor=(N,1)
        distance_tensor=torch.zeros((len(nodes),1), dtype=torch.float32) # distance_tensor=(N,1)

        Data_Processor.convert_edge_attr_to_float(graph)

        predecessor_dic, distance_dic = nx.bellman_ford_predecessor_and_distance(G=graph, source=source_id, weight='weight')
        predecessor_dic[source_id]=[source_id] # source_id: [] 인 공간 채우기 

        print(predecessor_dic)

        for tar in nodes:
            predecessor_tensor[tar][0]=predecessor_dic[tar][0]
            distance_tensor[tar][0]=distance_dic[tar]
        return predecessor_tensor,distance_tensor

    @staticmethod
    def compute_bellman_ford_step(graph,init=False,source_id=0):
        copy_graph=copy.deepcopy(graph)
        step_predecessor_label=torch.zeros((len(graph.nodes()),1),dtype=torch.float32) # step_predecessor_label=(N,1)
        step_x_label=torch.zeros((len(graph.nodes()),1),dtype=torch.float32) # step_x_label=(N,1)
        for node_idx in graph.nodes():
            step_x_label[node_idx][0]=graph.nodes[node_idx]['x'][0] 
            step_predecessor_label[node_idx][0]=graph.nodes[node_idx]['p'][0]
        
        if init:
            Data_Processor.convert_edge_attr_to_float(copy_graph)
            _, distance_dic=nx.bellman_ford_predecessor_and_distance(G=copy_graph, source=source_id, weight='weight')
            longest_shortest_path_length=max(distance_dic.values())
            copy_graph.graph['longest_distance']=longest_shortest_path_length+1

            # node들의 x feature 값들을 longest_shortest_path_length+1 값으로 초기화
            for node_idx in graph.nodes():
                copy_graph.nodes[node_idx]['x'][0]=copy_graph.graph['longest_distance']
                step_x_label[node_idx][0]=copy_graph.graph['longest_distance']

            # source node의 x, predecessor 값들을 초기화
            copy_graph.nodes[source_id]['x'][0]=0.0
            step_x_label[source_id][0]=0.0

            return copy_graph, step_x_label, step_predecessor_label

        for node_idx in graph.nodes():
            current_length=graph.nodes[node_idx]['x'][0] # 현재 노드의 최단 경로 길이
            prev=-1 # 이전 (선행)노드를 나타냄
            for neighbor_idx in graph.neighbors(node_idx):
                edge_data = graph.get_edge_data(node_idx, neighbor_idx)
                edge_weight = edge_data['edge_attr'][0]
                if graph.nodes[neighbor_idx]['x'][0] + edge_weight < current_length:
                    current_length = min(current_length, graph.nodes[neighbor_idx]['x'][0] + edge_weight)
                    prev = neighbor_idx
            if current_length < graph.nodes[node_idx]['x'][0]:
                copy_graph.nodes[node_idx]['x'][0] = current_length
                copy_graph.nodes[node_idx]['p'][0] = prev

        return copy_graph, step_predecessor_label, step_x_label

class Data_Analysis:
    @staticmethod
    def get_reachability_ratio(graph: nx.Graph):
        ### graph 내에서 reachability 비율이 50%에 가까운 상위 10개의 node 정보 반환

        reachability_data = []
        for source in graph.nodes():
            reachable_nodes = nx.single_source_shortest_path_length(graph, source).keys() # source 노드에서 도달 가능한 노드 집합 계산 (무방향 그래프)
            total_nodes = len(graph)
            if total_nodes > 0:
                reachability_ratio = len(reachable_nodes)/total_nodes # reachability 비율 계산
            else:
                reachability_ratio = 0.0
            reachability_data.append((source, reachability_ratio))

        reachability_df = pd.DataFrame(reachability_data, columns=['source', 'reachability_ratio'])

        # 도달 가능성 비율이 50%에 가까운 source 상위 10개 선택
        reachability_df['distance_to_50'] = np.abs(reachability_df['reachability_ratio'] - 0.5)
        top_10_sources = reachability_df.nsmallest(10, 'distance_to_50')[['source', 'reachability_ratio']] 

        result_dict = dict(zip(top_10_sources['source'], top_10_sources['reachability_ratio']))

        return result_dict