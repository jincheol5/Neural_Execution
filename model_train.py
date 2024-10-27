import os
import random
from tqdm import tqdm
import torch
from torch.nn import BCEWithLogitsLoss
from torch_geometric.utils.convert import from_networkx
from utils import Data_Processor



class Model_Trainer:
    def __init__(self,model=None):
        self.model=model
        self.dp=Data_Processor()
    
    def set_model(self,model):
        self.model=model
    
    def train_bfs(self,train_graph_list,hidden_dim=32,lr=0.01,epochs=10):
        optimizer=torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = BCEWithLogitsLoss()
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)

        for epoch in tqdm(range(epochs),desc="Training..."):
            for train_graph in train_graph_list:
                data=from_networkx(train_graph) # nx graph to pyg Data
                num_nodes=data.x.size(0)

                x=data.x
                edge_index=data.edge_index
                edge_attr=data.edge_attr
                x=x.to(device)
                edge_index=edge_index.to(device)
                edge_attr=edge_attr.to(device)

                t=0
                source_id=random.randint(0, num_nodes - 1)
                h=torch.zeros((num_nodes,hidden_dim), dtype=torch.float32) # h=(num_nodes,hidden_dim)
                h=h.detach() # 값만 전달하기 위해
                h=h.to(device)
                graph_t,x_t=self.dp.compute_bfs_step(graph=train_graph,source_id=source_id,init=True)
                x_t=x_t.to(device) # x_t=(num_nodes,1)
                
                while t < num_nodes:
                    graph_t,x_t=self.dp.compute_bfs_step(graph=graph_t,source_id=source_id)
                    optimizer.zero_grad()
                    output=self.model(x=x,edge_index=edge_index,edge_attr=edge_attr,pre_h=h)
                    # get and set h
                    h=output['h'] # h=(num_nodes,hidden_dim)
                    h=h.detach() # 값만 전달하기 위해
                    # get y, tau
                    y=output['y'] # y=(num_nodes,1)
                    tau=output['tau'] # tau=실수 값
                    if tau<=0.5:
                        break
                    # 오류 역전파 수행
                    loss=criterion(y,x_t)
                    loss.backward()
                    optimizer.step()

                    # set x, y값의 일부만 다음 time step의 x 값으로 전달


                    t+=1