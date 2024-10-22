import torch
import numpy as np
import networkx as nx 
import numpy as np
from collections import deque

def generate_ladder(n_nodes):
    graph = nx.ladder_graph(n_nodes)
    return graph

def breadth_first_search(graph, root=0):
    A = nx.to_numpy_array(graph,nodelist=sorted(graph.nodes()))
    A = np.array(A)
    nb_nodes = graph.number_of_nodes()
    x = np.zeros((nb_nodes))
    # Set the root node to 1 in the x vector to start the search from the root node
    x[root] = 1
    history = [x.copy()]
    q = deque()
    q.append(root)
    memory = set()
    terminate = False
    while len(q) > 0 and np.sum(x) < len(x):
        second_queue = deque()
        # Loop through all nodes in the current level of the search
        while len(q) > 0 and np.sum(x) < len(x):
            cur = q.popleft()
            memory.add(cur)
            neighbours = np.where(A[cur] > 0)[0]
            for n in neighbours:
                if n not in memory:
                    second_queue.append(n)
                    x[n] = 1
        # If all the nodes in the current level have been visited, add the current x vector to the history list
        if (x == history[-1]).all():
            terminate = True
            break
        history.append(x.copy())
        q = second_queue
    # Return the history list as a NumPy array
    return np.asarray(history)

# graph=nx.DiGraph()
# edge_index=[(0,1),(0,4),(1,3),(2,4),(3,2)]
# graph.add_edges_from(edge_index)


graph=generate_ladder(n_nodes=20)
# print(graph.edges)
history=breadth_first_search(graph=graph)

print(history)