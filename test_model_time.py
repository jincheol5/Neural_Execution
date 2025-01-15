import time
from tqdm import tqdm
from utils import Data_Loader,Metrics
from model_train import Execution_Engine
from model import NGAE_MPNN_BFS
import torch

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

graph_list_dict=Data_Loader.load_from_pickle(file_name="test_graph_1000_list_dict")
graph=graph_list_dict['erdos_renyi'][0]

model=NGAE_MPNN_BFS(x_dim=1,e_dim=1,latent_dim=32,aggr='max')
model.eval()
model.to(device)

start_time=time.time() 
for src in tqdm(graph.nodes(),desc="check time..."):
    Execution_Engine.initialize(graph=graph)
    Execution_Engine.set_source_node_feature(graph=graph,source_id=src)
    x=Execution_Engine.get_node_feature_to_tensor(graph=graph)
    h=torch.zeros((graph.number_of_nodes(),32), dtype=torch.float32)
    e=Execution_Engine.get_edge_feature_to_tensor(graph=graph)
    edge_index=Execution_Engine.get_edge_index(graph=graph)

    step=0
    while step<graph.number_of_nodes():
        step+=1

        x=x.to(device)
        h=h.to(device)
        e=e.to(device)
        edge_index=edge_index.to(device)

        output=model.forward(x=x,pre_h=h,edge_index=edge_index,edge_attr=e)
        y=output['y'] # [N,1], logit
        x=Metrics.compute_BFS_from_logit(logit=y)
        h=output['h']
end_time=time.time()
print(f"Execution time: {end_time-start_time:.5f} seconds")

