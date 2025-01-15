import time
from utils import Data_Loader,Graph_Algorithm
from model_train import Model_Trainer,Execution_Engine

graph_list_dict=Data_Loader.load_from_pickle(file_name="test_graph_1000_list_dict")
graph=graph_list_dict['erdos_renyi'][0]

start_time=time.time() 
for src in graph.nodes():
    Execution_Engine.initialize(graph=graph)
    Execution_Engine.set_source_node_feature(graph=graph,source_id=src)
    result=Graph_Algorithm.compute_reachability(graph=graph)
end_time=time.time()
print(f"Execution time: {end_time-start_time:.5f} seconds")