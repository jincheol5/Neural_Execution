import os
import random
import copy
from tqdm import tqdm
import numpy as np
from utils import Metrics,Graph_Algorithm
from model import NGAE_BFS
import torch
from torch_geometric.utils.convert import from_networkx

"""
<< node feature >>
r: temporal reachability
x: node feature => r

<< edge feature >>
w: edge weight
e: edge feature => w
"""
class Execution_Engine:
    """
    Execution Engine
    -manage input/output of the model
    -manage the repetitive execution of the model
    """
    @staticmethod
    def initialize(graph):
        # initialize node feature
        for node in graph.nodes():
            graph.nodes[node]['r']=0.0

    @staticmethod
    def set_source_node_feature(graph,source_id=0):
        # set source node feature
        graph.nodes[source_id]['r']=1.0

    @staticmethod
    def update_node_feature(graph,r):
        # set node r feature
        for node in graph.nodes():
            graph.nodes[node]['r']=r[node]
    
    @staticmethod
    def get_node_feature_to_tensor(graph):
        data=from_networkx(G=graph,group_node_attrs=['r'])
        return data.x # [N,1]

    @staticmethod
    def get_edge_feature_to_tensor(graph):
        data=from_networkx(G=graph,group_edge_attrs=['w'])
        return data.edge_attr # [E,1]

    @staticmethod
    def get_edge_index(graph):
        data=from_networkx(G=graph)
        return data.edge_index

class Model_Trainer:
    @staticmethod
    def train(model,train_graph_list_dict,val_graph_list_dict,latent_dim=32,lr=0.0005,epochs=10):
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        optimizer=torch.optim.Adam(model.parameters(),lr=lr)
        best_last_y_acc=0.0
        patience=0
        for epoch in tqdm(range(epochs),desc=f"Training BFS..."):
            model.train()
            model.to(device)
            for graph_type,graph_list in train_graph_list_dict.items():
                for graph in graph_list:
                    ### select source node
                    N=graph.number_of_nodes()
                    for source_id in graph.nodes():
                        ### initialize and set source of graph 
                        Execution_Engine.initialize(graph=graph)
                        Execution_Engine.set_source_node_feature(graph=graph,source_id=source_id)
                        Q=[source_id]

                        ### get first model input: x, edge_index, e
                        x=Execution_Engine.get_node_feature_to_tensor(graph=graph) # [N,1]
                        h=torch.zeros((N,latent_dim), dtype=torch.float32) # [N,latent_dim]
                        edge_index=Execution_Engine.get_edge_index(graph=graph) # [2,E]
                        e=Execution_Engine.get_edge_feature_to_tensor(graph=graph) # [E,1]

                        ### iterate model train
                        step=0
                        while step<N:
                            step+=1
                            ### move input to device
                            x=x.to(device)
                            h=h.to(device)
                            edge_index=edge_index.to(device)
                            e=e.to(device)

                            ### get step label
                            Q_next=Graph_Algorithm.compute_bfs_step(graph=graph,source_id=source_id,Q=Q) 
                            step_y_label=Execution_Engine.get_node_feature_to_tensor(graph=graph)
                            step_tau_label=Metrics.compute_tau_label(Q=Q_next)
                            step_y_label=step_y_label.to(device)
                            step_tau_label=step_tau_label.to(device)

                            ### get model output
                            output=model(x=x,pre_h=h,edge_index=edge_index,edge_attr=e)
                            y=output['y'] # [N,1], logit
                            h=output['h'].detach()
                            tau=output['tau'] # [1,1], logit

                            ### compute step loss and backpropagation
                            y_loss=Metrics.compute_BFS_loss(y,step_y_label)
                            tau_loss=Metrics.compute_tau_loss(tau,step_tau_label)
                            total_loss=y_loss+tau_loss
                            optimizer.zero_grad()
                            total_loss.backward()
                            optimizer.step()

                            ### terminate
                            if not Q_next:
                                break

                            ### set next input
                            x=Execution_Engine.get_node_feature_to_tensor(graph=graph) 
                            e=Execution_Engine.get_edge_feature_to_tensor(graph=graph) 

                            Q=Q_next
            print(f"{epoch+1} epoch training is finished.")
            ### check early stop
            step_y_acc,last_y_acc=Model_Trainer.validate(model=model,val_graph_list_dict=val_graph_list_dict,latent_dim=latent_dim)
            if best_last_y_acc<last_y_acc:
                best_last_y_acc=last_y_acc
                patience=0
            else:
                patience+=1
            print(f"Current best last p acc: {best_last_y_acc}")
            if patience>=10:
                print(f"Early stop, best last p acc: {best_last_y_acc}")
                break

    @staticmethod
    def validate(model,graph_list_dict,latent_dim=32):
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.eval()
        model.to(device)

        ### all_type_acc
        all_type_step_y_acc=[]
        all_type_last_y_acc=[]

        with torch.no_grad():
            for graph_type,graph_list in graph_list_dict.items():
                ### total_acc_list
                total_step_y_acc_list=[]
                total_last_y_acc_list=[]
                for graph in graph_list:
                    ### copy graph
                    execution_graph=copy.deepcopy(graph)

                    ### acc_list
                    step_y_acc_list=[]
                    last_y_acc_list=[]

                    ### select source node
                    N=graph.number_of_nodes()
                    for source_id in graph.nodes():
                        ### acc
                        step_y_acc=[]

                        ### initialize and set source of graph 
                        Execution_Engine.initialize(graph=graph)
                        Execution_Engine.set_source_node_feature(graph=graph,source_id=source_id)

                        Execution_Engine.initialize(graph=execution_graph)
                        Execution_Engine.set_source_node_feature(graph=execution_graph,source_id=source_id)
                        Q=[source_id]

                        ### get last tR, a, and predecessor label tensor=[N,1]
                        last_y_label=Graph_Algorithm.compute_reachability(graph=graph,source_id=source_id)
                        last_y_label=last_y_label.to(device)

                        ### get first model input: x, edge_index, e
                        x=Execution_Engine.get_node_feature_to_tensor(graph=execution_graph) # [N,1]
                        pre_y=x
                        h=torch.zeros((N,latent_dim), dtype=torch.float32) # [N,latent_dim]
                        edge_index=Execution_Engine.get_edge_index(graph=execution_graph) # [2,E]
                        e=Execution_Engine.get_edge_feature_to_tensor(graph=execution_graph) # [E,1]

                        ### iterate model train
                        step=0
                        while step<N:
                            step+=1
                            ### move input to device
                            x=x.to(device)
                            h=h.to(device)
                            edge_index=edge_index.to(device)
                            e=e.to(device)

                            ### get step label
                            Q_next=Graph_Algorithm.compute_bfs_step(graph=graph,source_id=source_id,Q=Q) 
                            step_y_label=Execution_Engine.get_node_feature_to_tensor(graph=graph)
                            step_tau_label=Metrics.compute_tau_label(Q=Q_next)
                            step_y_label=step_y_label.to(device)
                            step_tau_label=step_tau_label.to(device)

                            ### get model output
                            output=model(x=x,pre_h=h,edge_index=edge_index,edge_attr=e)
                            y=output['y'] # [N,1], logit
                            h=output['h'].detach()
                            tau=output['tau'] # [1,1], logit
                            y=Metrics.compute_BFS_from_logit(logit=y)
                            tau=Metrics.compute_tau_from_logit(logit=tau)

                            ### compute step acc
                            step_y_acc.append(Metrics.compute_BFS_accuracy(predict=y,label=step_y_label))

                            ### terminate
                            if torch.equal(pre_y,y):
                                break
                            if tau.item()<0.5:
                                break

                            ### set next input
                            x=Execution_Engine.get_node_feature_to_tensor(graph=execution_graph) 
                            e=Execution_Engine.get_edge_feature_to_tensor(graph=execution_graph) 

                            pre_y=y
                            Q=Q_next
                        ### compute mean of step acc and last acc
                        step_y_acc_list.append(np.mean(step_y_acc))
                        last_y_acc_list.append(Metrics.compute_BFS_accuracy(predict=y,label=last_y_label)) # compute last y and append to acc list
                    ### compute total acc
                    total_step_y_acc_list.append(np.mean(step_y_acc_list))
                    total_last_y_acc_list.append(np.mean(last_y_acc_list))
                ### compute final acc
                print(f"Validate {graph_type} type graph step y acc avg: {np.mean(total_step_y_acc_list):.2%} and last y acc avg: {np.mean(total_last_y_acc_list):.2%}")

                ### add to all_type_acc
                all_type_step_y_acc.append(np.mean(total_step_y_acc_list))
                all_type_last_y_acc.append(np.mean(total_last_y_acc_list))
        ### compute final_acc
        final_step_y_acc=np.mean(all_type_step_y_acc)
        final_last_y_acc=np.mean(all_type_last_y_acc)
        return final_step_y_acc,final_last_y_acc
    
    @staticmethod
    def evaluate(model_file_name,graph_list_dict,model_type='mpnn_max',latent_dim=32):
        ### Load model parameter
        match model_type:
            case 'gat':
                pass
            case 'mpnn_max':
                model=NGAE_BFS(x_dim=1,e_dim=1,latent_dim=latent_dim,aggr="max")
            case 'mpnn_min':
                model=NGAE_BFS(x_dim=1,e_dim=1,latent_dim=latent_dim,aggr="max")
            case 'mpnn_avg':
                model=NGAE_BFS(x_dim=1,e_dim=1,latent_dim=latent_dim,aggr="max")
        load_path=os.path.join(os.getcwd(),"inference",model_file_name+".pt")
        model.load_state_dict(torch.load(load_path))
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.eval()
        model.to(device)
        with torch.no_grad():
            for graph_type,graph_list in tqdm(graph_list_dict.items(),desc=f"Evaluating BFS..."):
                ### total_acc_list
                total_step_y_acc_list=[]
                total_last_y_acc_list=[]
                for graph in graph_list:
                    ### copy graph
                    execution_graph=copy.deepcopy(graph)

                    ### acc_list
                    step_y_acc_list=[]
                    last_y_acc_list=[]

                    ### select source node
                    N=graph.number_of_nodes()
                    for source_id in graph.nodes():
                        ### acc
                        step_y_acc=[]

                        ### initialize and set source of graph 
                        Execution_Engine.initialize(graph=graph)
                        Execution_Engine.set_source_node_feature(graph=graph,source_id=source_id)

                        Execution_Engine.initialize(graph=execution_graph)
                        Execution_Engine.set_source_node_feature(graph=execution_graph,source_id=source_id)
                        Q=[source_id]

                        ### get last tR, a, and predecessor label tensor=[N,1]
                        last_y_label=Graph_Algorithm.compute_reachability(graph=graph,source_id=source_id)
                        last_y_label=last_y_label.to(device)

                        ### get first model input: x, edge_index, e
                        x=Execution_Engine.get_node_feature_to_tensor(graph=execution_graph) # [N,1]
                        pre_y=x
                        h=torch.zeros((N,latent_dim), dtype=torch.float32) # [N,latent_dim]
                        edge_index=Execution_Engine.get_edge_index(graph=execution_graph) # [2,E]
                        e=Execution_Engine.get_edge_feature_to_tensor(graph=execution_graph) # [E,1]

                        ### iterate model train
                        step=0
                        while step<N:
                            step+=1
                            ### move input to device
                            x=x.to(device)
                            h=h.to(device)
                            edge_index=edge_index.to(device)
                            e=e.to(device)

                            ### get step label
                            Q_next=Graph_Algorithm.compute_bfs_step(graph=graph,source_id=source_id,Q=Q) 
                            step_y_label=Execution_Engine.get_node_feature_to_tensor(graph=graph)
                            step_tau_label=Metrics.compute_tau_label(Q=Q_next)
                            step_y_label=step_y_label.to(device)
                            step_tau_label=step_tau_label.to(device)

                            ### get model output
                            output=model(x=x,pre_h=h,edge_index=edge_index,edge_attr=e)
                            y=output['y'] # [N,1], logit
                            h=output['h'].detach()
                            tau=output['tau'] # [1,1], logit
                            y=Metrics.compute_BFS_from_logit(logit=y)
                            tau=Metrics.compute_tau_from_logit(logit=tau)

                            ### compute step acc
                            step_y_acc.append(Metrics.compute_BFS_accuracy(predict=y,label=step_y_label))

                            ### terminate
                            if torch.equal(pre_y,y):
                                break
                            if tau.item()<0.5:
                                break

                            ### set next input
                            x=Execution_Engine.get_node_feature_to_tensor(graph=execution_graph) 
                            e=Execution_Engine.get_edge_feature_to_tensor(graph=execution_graph) 

                            pre_y=y
                            Q=Q_next
                        ### compute mean of step acc and last acc
                        step_y_acc_list.append(np.mean(step_y_acc))
                        last_y_acc_list.append(Metrics.compute_BFS_accuracy(predict=y,label=last_y_label)) # compute last y and append to acc list
                    ### compute total acc
                    total_step_y_acc_list.append(np.mean(step_y_acc_list))
                    total_last_y_acc_list.append(np.mean(last_y_acc_list))
                ### compute final acc
                print(f"Evaluate {graph_type} type graph step y acc avg: {np.mean(total_step_y_acc_list):.2%} and last y acc avg: {np.mean(total_last_y_acc_list):.2%}")