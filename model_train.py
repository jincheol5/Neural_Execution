import os
import random
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss,MSELoss
from torch_geometric.utils.convert import from_networkx
from utils import Data_Processor
from model import BFS_Neural_Execution,BF_Distance_Neural_Execution



class Model_Trainer:
    @staticmethod
    def compute_bfs_accuracy(y,label):
        # y, label은 gpu 위에 있으나, 결과값 acc는 cpu로 반환
        N=y.size(0)
        correct=(y == label).sum().item()
        acc=correct/N
        return acc
    
    @staticmethod
    def compute_bf_distance_accuracy(dist,dist_label):
        # comput distance acc
        # 절대 오차가 delta 이하인 경우 정확한 예측으로 간주
        delta=0.5
        correct_predictions = (torch.abs(dist - dist_label) < delta).float()
        dist_acc = correct_predictions.mean().item()

        return dist_acc

    @staticmethod
    def compare_tensors(tensor1,tensor2):
        if torch.equal(tensor1, tensor2):
            # 값이 동일하면 (1, 1) 형태의 0.0 텐서 반환
            return torch.tensor([[0.0]])
        else:
            # 값이 다르면 (1, 1) 형태의 1.0 텐서 반환
            return torch.tensor([[1.0]])
    
    @staticmethod
    def save_model_state_dict(model,model_name="model_parameter"):
        file_name=model_name+".pt"
        save_path=os.path.join(os.getcwd(),"inference",file_name)
        torch.save(model.state_dict(),save_path)
    
    @staticmethod
    def train_bfs(model,train_graph_list_dict,val_graph_list_dict,hidden_dim=32,lr=0.0005,epochs=10):
        optimizer=torch.optim.Adam(model.parameters(), lr=lr)
        criterion = BCEWithLogitsLoss()
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        for epoch in tqdm(range(epochs),desc="Training..."):
            model.train()
            for _,train_graph_list  in train_graph_list_dict.items():
                for train_graph in train_graph_list:
                    data=from_networkx(train_graph) # nx graph to pyg Data
                    N=data.x.size(0)
                    edge_index=data.edge_index.to(device)
                    edge_attr=data.edge_attr.to(device)

                    source_id=random.randint(0, N - 1)

                    # compute last x label
                    last_x_label=Data_Processor.compute_reachability(graph=train_graph,source_id=source_id).to(device)

                    # initialize step
                    h_0=torch.zeros((N,hidden_dim), dtype=torch.float32) # h_0=(N,hidden_dim)
                    graph_0,x_0=Data_Processor.compute_bfs_step(graph=train_graph,source_id=source_id,init=True)
                    h=h_0.to(device)
                    x=x_0.to(device)
                    graph=graph_0

                    t=0
                    while t < N:
                        optimizer.zero_grad()
                        graph_t,x_t=Data_Processor.compute_bfs_step(graph=graph,source_id=source_id)
                        x_t=x_t.to(device)

                        # get model output
                        output=model(x=x,edge_index=edge_index,edge_attr=edge_attr,pre_h=h)
                        h=output['h'].detach() # h=(N,hidden_dim)
                        y=output['y'] # y=(N,1)
                        tau=output['tau'] # tau=(1,1)

                        # set terminate label at t 
                        tau_t=Model_Trainer.compare_tensors(tensor1=last_x_label,tensor2=x_t).to(device) # 마지막 step이면 0.0 아니면 1.0

                        # 손실 함수 계산 및 오류 역전파 수행
                        x_loss=criterion(y,x_t)
                        terminate_loss=criterion(tau,tau_t)
                        total_loss=x_loss+terminate_loss
                        total_loss.backward()
                        optimizer.step()

                        # 마지막 step인 경우 종료
                        if tau_t.item()==0.0:
                            break
                        graph=graph_t
                        x=x_t
                        t+=1
            print(f"{epoch+1} epoch training is finished.")
            Model_Trainer.validate_bfs(model=model,val_graph_list_dict=val_graph_list_dict,hidden_dim=32)

    @staticmethod
    def validate_bfs(model,val_graph_list_dict,hidden_dim=32):
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        
        with torch.no_grad():
            for val_graph_type,val_graph_list in val_graph_list_dict.items():
                step_x_acc_avg_list=[]
                last_x_acc_list=[]
                for val_graph in val_graph_list:
                    step_x_acc_list=[]

                    data=from_networkx(val_graph) # nx graph to pyg Data
                    N=data.x.size(0)
                    edge_index=data.edge_index.to(device)
                    edge_attr=data.edge_attr.to(device)

                    source_id=random.randint(0, N - 1)

                    h=torch.zeros((N,hidden_dim), dtype=torch.float32).to(device) # h=(N,hidden_dim)
                    last_x_label=Data_Processor.compute_reachability(graph=val_graph,source_id=source_id)
                    last_x_label=last_x_label.to(device)

                    # initialize step
                    graph_0,x_0=Data_Processor.compute_bfs_step(graph=val_graph,source_id=source_id,init=True)
                    x=x_0.to(device) 
                    graph=graph_0

                    t=0
                    while t < N:
                        graph_t,x_t=Data_Processor.compute_bfs_step(graph=graph,source_id=source_id)
                        x_t=x_t.to(device)

                        # get model output
                        output=model(x=x,edge_index=edge_index,edge_attr=edge_attr,pre_h=h)
                        # get and set h
                        h=output['h'] # h=(N,hidden_dim)
                        # get y, tau
                        y=output['y'] # y=(N,1)
                        cls_y=(y > 0.5).float() # y 값을 1.0 or 0.0으로 변환, GPU로 유지
                        tau=output['tau'] # tau=(1,1)
                        
                        # compute step accuracy
                        step_x_acc=Model_Trainer.compute_bfs_accuracy(y=cls_y,label=x_t)
                        step_x_acc_list.append(step_x_acc)

                        # set graph and x
                        graph=graph_t
                        x=cls_y

                        # terminate
                        if tau.item()<=0.5:
                            break
                        t+=1

                    # compute step acc and last acc
                    step_x_acc_avg=np.mean(step_x_acc_list)
                    last_x_acc=Model_Trainer.compute_bfs_accuracy(y=x,label=last_x_label)
                    step_x_acc_avg_list.append(step_x_acc_avg)
                    last_x_acc_list.append(last_x_acc)

                print(f"Validate {val_graph_type} graph step_acc_avg: {np.mean(step_x_acc_avg_list):.2%} and last_acc_avg: {np.mean(last_x_acc_list):.2%}")
                print()

    @staticmethod
    def evaluate_bfs(test_graph_list_dict,model_file_name,hidden_dim=32):
        model=BFS_Neural_Execution(hidden_dim=hidden_dim)
        load_path=os.path.join(os.getcwd(), "inference",model_file_name+".pt")
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.load_state_dict(torch.load(load_path))
        model.eval()

        with torch.no_grad():
            for test_graph_type,test_graph_list in test_graph_list_dict.items():
                step_x_acc_avg_list=[]
                last_x_acc_list=[]
                for test_graph in tqdm(test_graph_list,desc="Evaluating "+test_graph_type+" graph..."):
                    step_x_acc_list=[]

                    data=from_networkx(test_graph) # nx graph to pyg Data
                    N=data.x.size(0)
                    edge_index=data.edge_index.to(device)
                    edge_attr=data.edge_attr.to(device)

                    source_id=random.randint(0, N - 1)

                    h=torch.zeros((N,hidden_dim), dtype=torch.float32).to(device) # h=(N,hidden_dim)
                    last_x_label=Data_Processor.compute_reachability(graph=test_graph,source_id=source_id)
                    last_x_label=last_x_label.to(device)

                    # initialize step
                    graph_0,x_0=Data_Processor.compute_bfs_step(graph=test_graph,source_id=source_id,init=True)
                    x=x_0.to(device) 
                    graph=graph_0

                    t=0
                    while t < N:
                        graph_t,x_t=Data_Processor.compute_bfs_step(graph=graph,source_id=source_id)
                        x_t=x_t.to(device)

                        # get model output
                        output=model(x=x,edge_index=edge_index,edge_attr=edge_attr,pre_h=h)
                        # get and set h
                        h=output['h'] # h=(N,hidden_dim)
                        # get y, ter
                        y=output['y'] # y=(N,1)
                        cls_y=(y > 0.5).float() # y 값을 1.0 or 0.0으로 변환, GPU로 유지
                        tau=output['tau'] # tau=(1,1)
                        
                        # compute step accuracy
                        step_x_acc=Model_Trainer.compute_bfs_accuracy(y=cls_y,label=x_t)
                        step_x_acc_list.append(step_x_acc)

                        # set graph and x
                        graph=graph_t
                        x=cls_y

                        # terminate
                        if tau.item()<=0.5:
                            break
                        t+=1

                    # compute step acc and last acc
                    step_x_acc_avg=np.mean(step_x_acc_list)
                    last_x_acc=Model_Trainer.compute_bfs_accuracy(y=x,label=last_x_label)
                    step_x_acc_avg_list.append(step_x_acc_avg)
                    last_x_acc_list.append(last_x_acc)

                print(f"Evaluate {test_graph_type} graph step_acc_avg: {np.mean(step_x_acc_avg_list):.2%} and last_acc_avg: {np.mean(last_x_acc_list):.2%}")
                print()

    @staticmethod
    def train_bf_distance(model,train_graph_list_dict,val_graph_list_dict,hidden_dim=32,lr=0.0005,epochs=10):
        optimizer=torch.optim.Adam(model.parameters(), lr=lr)
        dist_criterion = MSELoss()
        tau_criterion=BCEWithLogitsLoss()
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        for epoch in tqdm(range(epochs),desc="Training..."):
            model.train()
            for _,train_graph_list  in train_graph_list_dict.items():
                for train_graph in train_graph_list:
                    data=from_networkx(train_graph) # nx graph to pyg Data
                    N=data.x.size(0)
                    edge_index=data.edge_index.to(device)
                    edge_attr=data.edge_attr.to(device)

                    source_id=random.randint(0, N - 1)

                    # compute last x label
                    _,last_x_label=Data_Processor.compute_shortest_path_and_predecessor(graph=train_graph,source_id=source_id)
                    last_x_label=last_x_label.to(device)

                    # initialize step
                    h_0=torch.zeros((N,hidden_dim), dtype=torch.float32) # h_0=(N,hidden_dim)
                    graph_0,_,x_0=Data_Processor.compute_bellman_ford_step(graph=train_graph,source_id=source_id,init=True)
                    h=h_0.to(device)
                    x=x_0.to(device)
                    graph=graph_0

                    t=0
                    while t < N:
                        optimizer.zero_grad()
                        graph_t,_,x_t=Data_Processor.compute_bellman_ford_step(graph=graph,source_id=source_id)
                        x_t=x_t.to(device)

                        # get model output
                        output=model(x=x,edge_index=edge_index,edge_attr=edge_attr,pre_h=h)
                        h=output['h'].detach() # h=(N,hidden_dim)
                        dist=output['dist'] # dist=(N,1)
                        tau=output['tau'] # tau=(1,1)

                        # set terminate label at t 
                        tau_t=Model_Trainer.compare_tensors(tensor1=last_x_label,tensor2=x_t).to(device) # 마지막 step이면 0.0 아니면 1.0

                        # 손실 함수 계산 및 오류 역전파 수행
                        dist_loss=dist_criterion(dist,x_t)
                        terminate_loss=tau_criterion(tau,tau_t)
                        total_loss=dist_loss+terminate_loss
                        total_loss.backward()
                        optimizer.step()

                        # 마지막 step인 경우 종료
                        if tau_t.item()==0.0:
                            break
                        graph=graph_t
                        x=x_t
                        t+=1
            print(f"{epoch+1} epoch training is finished.")
            Model_Trainer.validate_bf_distance(model=model,val_graph_list_dict=val_graph_list_dict,hidden_dim=32)

    @staticmethod
    def validate_bf_distance(model,val_graph_list_dict,hidden_dim=32):
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        
        with torch.no_grad():
            for val_graph_type,val_graph_list in val_graph_list_dict.items():
                step_x_acc_avg_list=[]
                last_x_acc_list=[]
                for val_graph in val_graph_list:
                    step_x_acc_list=[]

                    data=from_networkx(val_graph) # nx graph to pyg Data
                    N=data.x.size(0)
                    edge_index=data.edge_index.to(device)
                    edge_attr=data.edge_attr.to(device)

                    source_id=random.randint(0, N - 1)

                    h=torch.zeros((N,hidden_dim), dtype=torch.float32).to(device) # h=(N,hidden_dim)
                    _,last_x_label=Data_Processor.compute_shortest_path_and_predecessor(graph=val_graph,source_id=source_id)
                    last_x_label=last_x_label.to(device)

                    # initialize step
                    graph_0,_,x_0=Data_Processor.compute_bellman_ford_step(graph=val_graph,source_id=source_id,init=True)
                    x=x_0.to(device) 
                    graph=graph_0

                    t=0
                    while t < N:
                        graph_t,_,x_t=Data_Processor.compute_bellman_ford_step(graph=graph,source_id=source_id)
                        x_t=x_t.to(device)

                        # get model output
                        output=model(x=x,edge_index=edge_index,edge_attr=edge_attr,pre_h=h)
                        h=output['h'] # h=(N,hidden_dim)
                        dist=output['dist'] # dist=(N,1)
                        tau=output['tau'] # tau=(1,1)

                        # compute step accuracy
                        step_x_acc=Model_Trainer.compute_bf_distance_accuracy(dist=dist,dist_label=x_t)
                        step_x_acc_list.append(step_x_acc)

                        # set graph and x
                        graph=graph_t
                        x=dist

                        # terminate
                        if tau.item()<=0.5:
                            break
                        t+=1

                    # compute step acc and last acc
                    step_x_acc_avg=np.mean(step_x_acc_list)
                    last_x_acc=Model_Trainer.compute_bf_distance_accuracy(y=x,label=last_x_label)
                    step_x_acc_avg_list.append(step_x_acc_avg)
                    last_x_acc_list.append(last_x_acc)

                print(f"Validate {val_graph_type} graph step_acc_avg: {np.mean(step_x_acc_avg_list):.2%} and last_acc_avg: {np.mean(last_x_acc_list):.2%}")
                print()

    @staticmethod
    def evaluate_bf_distance(test_graph_list_dict,model_file_name,hidden_dim=32):
        model=BF_Distance_Neural_Execution(hidden_dim=hidden_dim)
        load_path=os.path.join(os.getcwd(), "inference",model_file_name+".pt")
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.load_state_dict(torch.load(load_path))
        model.eval()

        with torch.no_grad():
            for test_graph_type,test_graph_list in test_graph_list_dict.items():
                step_x_acc_avg_list=[]
                last_x_acc_list=[]
                for test_graph in tqdm(test_graph_list,desc="Evaluating "+test_graph_type+" graph..."):
                    step_x_acc_list=[]

                    data=from_networkx(test_graph) # nx graph to pyg Data
                    N=data.x.size(0)
                    edge_index=data.edge_index.to(device)
                    edge_attr=data.edge_attr.to(device)

                    source_id=random.randint(0, N - 1)

                    h=torch.zeros((N,hidden_dim), dtype=torch.float32).to(device) # h=(N,hidden_dim)
                    _,last_x_label=Data_Processor.compute_shortest_path_and_predecessor(graph=test_graph,source_id=source_id)
                    last_x_label=last_x_label.to(device)

                    # initialize step
                    graph_0,_,x_0=Data_Processor.compute_bellman_ford_step(graph=test_graph,source_id=source_id,init=True)
                    x=x_0.to(device) 
                    graph=graph_0

                    t=0
                    while t < N:
                        graph_t,_,x_t=Data_Processor.compute_bellman_ford_step(graph=graph,source_id=source_id)
                        x_t=x_t.to(device)

                        # get model output
                        output=model(x=x,edge_index=edge_index,edge_attr=edge_attr,pre_h=h)
                        h=output['h'] # h=(N,hidden_dim)
                        dist=output['dist'] # y=(N,1)
                        tau=output['tau'] # tau=(1,1)

                        # compute step accuracy
                        step_x_acc=Model_Trainer.compute_bf_distance_accuracy(y=dist,label=x_t)
                        step_x_acc_list.append(step_x_acc)

                        # set graph and x
                        graph=graph_t
                        x=dist

                        # terminate
                        if tau.item()<=0.5:
                            break
                        t+=1

                    # compute step acc and last acc
                    step_x_acc_avg=np.mean(step_x_acc_list)
                    last_x_acc=Model_Trainer.compute_bf_distance_accuracy(y=dist,label=x_t)
                    step_x_acc_avg_list.append(step_x_acc_avg)
                    last_x_acc_list.append(last_x_acc)

                print(f"Evaluate {test_graph_type} graph step_acc_avg: {np.mean(step_x_acc_avg_list):.2%} and last_acc_avg: {np.mean(last_x_acc_list):.2%}")
                print()