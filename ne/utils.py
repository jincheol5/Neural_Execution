import os
import random
import math
import copy
import networkx as nx
import pandas as pd
import numpy as np
import torch
import torch_geometric as tg

def generate_self_edges(G):
    self_edges = [(node, node) for node in G.nodes()]
    G.add_edges_from(self_edges)
    return G

def generate_edge_weights(G):
    for u, v in G.edges():
        G[u][v]['weight'] = random.uniform(0.2, 1)
    return G


def generate_all_graph_types(N_graphs, N_nodes):
    exceptions = {50: (7, 7)}
    add = False

    graphs = []
    for g in range(N_graphs):
        # ladder graph
        n = int(N_nodes / 2)   # number of rungs = number of nodes / 2
        G = nx.ladder_graph(n)
        G = generate_self_edges(G)
        G = generate_edge_weights(G)
        graphs.append(G)

        # 2D grid graph
        n = int(math.ceil(math.sqrt(N_nodes))) # number of rows
        m = int(math.ceil(N_nodes / n)) # number of columns
        if m*n != N_nodes:
            n, m = exceptions[N_nodes]
            add = True
        G = nx.grid_2d_graph(n, m)
        if add:
            G.add_node((6, 7))
            G.add_edge((6, 6), (6, 7))
        add = False
        G = generate_self_edges(G)
        G = generate_edge_weights(G)
        graphs.append(G)

        # Erdos Renyi graph
        p = min(math.log(N_nodes, 2)/N_nodes, 0.5) # edge probability
        G = nx.erdos_renyi_graph(N_nodes, p)
        G = generate_self_edges(G)
        G = generate_edge_weights(G)
        graphs.append(G)

    return graphs

def bfs_step(graph, init=False, source_node=None, label_name='bfs_label'):
    G_copy = copy.deepcopy(graph)
    if init:
        for node in graph.nodes():
            G_copy.nodes[node][label_name] = 0.
    # choosing the source node
        if source_node is None:
            nodes = list(graph.nodes())
            index = random.randint(0, len(nodes) - 1)
            source_node = nodes[index]
        else:
            source_node = source_node
        G_copy.graph['source_node'] = source_node
        # setting the source node
        G_copy.nodes[source_node][label_name] = 1.
        return G_copy

    for node in graph.nodes():
        if graph.nodes[node][label_name] == 1:
            for neighbor in graph.neighbors(node):
                if graph.nodes[neighbor][label_name] == 0:
                    G_copy.nodes[neighbor][label_name] = 1.
    return G_copy

def bellman_ford_step(graph, init=False, source_node=None, label_name='bf_label', predecessor_label='pred'): # TODO mozda je zapravo svatko sam sebi predecessor na pocetku cisto po formuli
    # To provide a numerically stable value for +inf, such entries are set to 
    # the lenght of the longest shortest path in the graph + 1

    G_copy = copy.deepcopy(graph)
    nodes = list(graph.nodes())
    if init:
        # choosing the source node
        if source_node is None:
            index = random.randint(0, len(nodes) - 1)
            source_node = nodes[index]
        else:
            source_node = source_node
        # calculating the longest shortest path
        path_lengths = nx.single_source_dijkstra_path_length(G_copy, source_node)
        longest_shortest_path = max(path_lengths.values())
        G_copy.graph['longest_shortest_path'] = longest_shortest_path + 1

        for i, node in enumerate(graph.nodes()):
            G_copy.nodes[node][label_name] = G_copy.graph['longest_shortest_path']
            # because of categorical cross entropy predecessor needs to be set on some node, so itself
            G_copy.nodes[node][predecessor_label] = i
    
        G_copy.nodes[source_node][label_name] = 0.
        G_copy.graph['source_node'] = source_node # predecessor label of the source_node already set on itself
        return G_copy

    for node in graph.nodes():
        current = graph.nodes[node][label_name] 
        pred = -1.
        for neighbor in graph.neighbors(node):
            edge_data = graph.get_edge_data(node, neighbor)
            weight = edge_data['weight']
            if graph.nodes[neighbor][label_name] + weight < current:
                current = min(current, graph.nodes[neighbor][label_name] + weight)
                pred = neighbor
        if current < graph.nodes[node][label_name]:
            G_copy.nodes[node][label_name] = current
            # setting predecessor
            i = nodes.index(pred)
            G_copy.nodes[node][predecessor_label] = i
    return G_copy

def prepare_initial_graph(G, source_node=None):
    G = bfs_step(G, init=True, source_node=source_node)
    # same source node for both algorithms
    source_node = G.graph['source_node']
    G = bellman_ford_step(G, init=True, source_node=source_node)
    return G


def node_match_bfs(G1_node, G2_node):
    # G1_node are all features for one node in G1 in a dictionary form
    if G1_node['bfs_label'] == G2_node['bfs_label']:
        return True
    else:
        return False

def node_match_bf(G1_node, G2_node):
    # G1_node are all features for one node in G1 in a dictionary form
    if G1_node['pred'] == G2_node['pred'] and math.isclose(G1_node['bf_label'], G2_node['bf_label'], rel_tol=1e-7):
        return True
    else:
        return False

def edge_match(G1_edge, G2_edge):
    # G1_edge are all features for one edge in G1 in a dictionary form
    if math.isclose(G1_edge['weight'], G2_edge['weight'], rel_tol=1e-7):
        return True
    else:
        return False

def prepare_pyg_data(G, hidden_dim=-1, h1=None, h2=None): 
    d = tg.utils.convert.from_networkx(G)
    d.edge_attr = d.weight
    if h1 is None or h2 is None:
        d.h_bfs = torch.zeros((d.num_nodes, hidden_dim))
        d.h_bf = torch.zeros((d.num_nodes, hidden_dim))
    else:
        d.h_bfs = h1.detach()
        d.h_bf = h2.detach()
    next_G = bfs_step(G)
    next_G = bellman_ford_step(next_G)
    d1 = tg.utils.convert.from_networkx(next_G) 
    # algoritham needs to continue further -> terminated set to 1 (when algorithm finishes, set to 0)
    bfs_terminated = 0. if nx.is_isomorphic(G, next_G, node_match=node_match_bfs, edge_match=edge_match) else 1.
    bf_terminated = 0. if nx.is_isomorphic(G, next_G, node_match=node_match_bf, edge_match=edge_match) else 1.
    # output labels are inputs of the next step
    d.y = [d1.bfs_label.reshape(-1, 1).clone(), d1.bf_label.reshape(-1, 1).clone(), d1.pred.clone().long(), torch.tensor([bfs_terminated]), torch.tensor([bf_terminated])] # TODO probably going to need to change for predecessor 
    return d, next_G # return new_graph as well so it can be used in next time step of the sequence

def prepare_pyg_data_for_bfs(G, hidden_dim=-1, h1=None): 
    d = tg.utils.convert.from_networkx(G)
    d.edge_attr = d.weight
    if h1 is None:
        d.h_bfs = torch.zeros((d.num_nodes, hidden_dim))
    else:
        d.h_bfs = h1.detach()
    next_G = bfs_step(G)
    #  next_G = bellman_ford_step(next_G)
    d1 = tg.utils.convert.from_networkx(next_G)
    # algoritham needs to continue further -> terminated set to 1 (when algorithm finishes, set to 0)
    bfs_terminated = 0. if nx.is_isomorphic(G, next_G, node_match=node_match_bfs, edge_match=edge_match) else 1.
    #  bf_terminated = 0. if nx.is_isomorphic(G, next_G, node_match=node_match_bf, edge_match=edge_match) else 1.
    # output labels are inputs of the next step
    d.y = [d1.bfs_label.reshape(-1, 1).clone(), torch.tensor([bfs_terminated])] # TODO probably going to need to change for predecessor 
    return d, next_G # return new_graph as well so it can be used in next time step of the sequence

def prepare_pyg_data_for_bf(G, hidden_dim=-1, h1=None): 
    d = tg.utils.convert.from_networkx(G)
    d.edge_attr = d.weight
    if h1 is None:
        d.h_bf = torch.zeros((d.num_nodes, hidden_dim))
    else:
        d.h_bf = h1.detach()
    next_G = bellman_ford_step(G)
    #  next_G = bfs_step(next_G)
    d1 = tg.utils.convert.from_networkx(next_G)
    # algoritham needs to continue further -> terminated set to 1 (when algorithm finishes, set to 0)
    #  bfs_terminated = 0. if nx.is_isomorphic(G, next_G, node_match=node_match_bfs, edge_match=edge_match) else 1.
    bf_terminated = 0. if nx.is_isomorphic(G, next_G, node_match=node_match_bf, edge_match=edge_match) else 1.
    # output labels are inputs of the next step
    d.y = [d1.bf_label.reshape(-1, 1).clone(), torch.tensor([bf_terminated]), d1.pred.clone().long()] # TODO probably going to need to change for predecessor 
    return d, next_G # return new_graph as well so it can be used in next time step of the sequence

def prepare_output(out, pred=False):
    # We have to split BFoutput because it is representing distance and predecessor, and
    # we need to prepare outputs - each has to be seperate so we can use appropriate loss 
    (bfs_out, bfs_term, bfs_h) = out[0]
    (bf_dist, bf_term, bf_h) = out[1]
    if pred:
        bf_pred = out[2]
    else:
        bf_pred = None
    return bfs_out, bf_dist, bf_pred, bfs_term, bf_term, bfs_h, bf_h # TODO check termination output

def prepare_output_one_alg(out, pred=False):
    # We have to split BFoutput because it is representing distance and predecessor, and
    # we need to prepare outputs - each has to be seperate so we can use appropriate loss 
    (alg_out, term, h) = out[0]
    if pred:
        pred = out[1]
        return alg_out, term, pred, h
    return alg_out, term, h 