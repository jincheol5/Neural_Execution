import os
import random
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss,CrossEntropyLoss,MSELoss
from torch_geometric.utils.convert import from_networkx
from utils import Data_Processor
from model import BFS_Neural_Execution,BF_Neural_Execution,BF_Distance_Neural_Execution



class Model_Trainer:
    def __init__(self,model=None):
        self.model=model
    
    def set_model(self,model):
        self.model=model
    
    def compute_bfs_accuracy(self,y,label):
        # y, label은 gpu 위에 있으나, 결과값 acc는 cpu로 반환
        N=y.size(0)
        y=y.squeeze() # y=(N,1)->(N,)
        label=label.squeeze() # label=(N,)->(N,)
        correct=(y == label).sum().item()
        acc=correct/N
        return acc
    
    def compute_bf_accuracy(self,prec,dist,prec_label,dist_label):
        # dist, prec, dist_label, prec_label은 gpu 위에 있으나, 결과값 dist_acc,prec_acc는 cpu로 반환
        N=dist.size(0)

        prec=prec.argmax(dim=1) # prec=(N,N)->(N,1)
        prec=prec.squeeze() # prec=(N,1)->(N,)
        prec_label=prec_label.squeeze() # prec_label=(N,1)->(N,)
        dist=dist.squeeze() # dist=(N,1)->(N,)
        dist_label=dist_label.squeeze() # dist_label=(N,1)->(N,)

        # compute predecessor acc
        prec_correct=(prec == prec_label).sum().item()
        prec_acc=prec_correct/N

        # comput distance acc
        # 절대 오차가 delta 이하인 경우 정확한 예측으로 간주
        delta=0.5
        correct_predictions = (torch.abs(dist - dist_label) < delta).float()
        dist_acc = correct_predictions.mean().item()

        return prec_acc, dist_acc
    
    def compute_bf_distance_accuracy(self,dist,dist_label):
        # dist, prec, dist_label, prec_label은 gpu 위에 있으나, 결과값 dist_acc,prec_acc는 cpu로 반환
        N=dist.size(0)

        dist=dist.squeeze() # dist=(N,1)->(N,)
        dist_label=dist_label.squeeze() # dist_label=(N,1)->(N,)

        # comput distance acc
        # 절대 오차가 delta 이하인 경우 정확한 예측으로 간주
        delta=0.5
        correct_predictions = (torch.abs(dist - dist_label) < delta).float()
        dist_acc = correct_predictions.mean().item()

        return dist_acc
    
    def compare_tensors(self,tensor1,tensor2):
        if torch.equal(tensor1, tensor2):
            # 값이 동일하면 (1, 1) 형태의 0.0 텐서 반환
            return torch.tensor([[0.0]])
        else:
            # 값이 다르면 (1, 1) 형태의 1.0 텐서 반환
            return torch.tensor([[1.0]])
    
    def save_model_state_dict(self,model_name="model_parameter"):
        file_name=model_name+".pt"
        save_path=os.path.join(os.getcwd(),"inference",file_name)
        torch.save(self.model.state_dict(),save_path)
    
    def train_bfs(self,train_graph_list,hidden_dim=32,lr=0.01,epochs=10):
        optimizer=torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = BCEWithLogitsLoss()
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        self.model.train()

        for epoch in tqdm(range(epochs),desc="Training..."):
            for train_graph in train_graph_list:
                data=from_networkx(train_graph) # nx graph to pyg Data
                N=data.x.size(0)
                edge_index=data.edge_index.to(device)
                edge_attr=data.edge_attr.to(device)

                source_id=random.randint(0, N - 1)

                h=torch.zeros((N,hidden_dim), dtype=torch.float32).detach().to(device) # h=(N,hidden_dim)
                last_x_label=Data_Processor.compute_reachability(graph=train_graph,source_id=source_id).to(device)

                # initialize step
                graph_0,x_0=Data_Processor.compute_bfs_step(graph=train_graph,source_id=source_id,init=True)
                x=x_0.detach().to(device) 
                graph_t=graph_0

                t=0
                while t < N:
                    optimizer.zero_grad()
                    graph_t,x_t=Data_Processor.compute_bfs_step(graph=graph_t,source_id=source_id)
                    x_t=x_t.to(device)

                    # get model output
                    output=self.model(x=x,edge_index=edge_index,edge_attr=edge_attr,pre_h=h)
                    h=output['h'].detach() # h=(N,hidden_dim)
                    y=output['y'] # y=(N,1)
                    tau=output['tau'] # tau=(1,1)

                    # set terminate label at t 
                    tau_t=self.compare_tensors(tensor1=last_x_label,tensor2=x_t).to(device) # 마지막 step이면 0.0 아니면 1.0

                    # 손실 함수 계산 및 오류 역전파 수행
                    x_loss=criterion(y.squeeze(),x_t.squeeze())
                    terminate_loss=criterion(tau.squeeze(),tau_t.squeeze())
                    total_loss=x_loss+terminate_loss
                    total_loss.backward()
                    optimizer.step()

                    # 마지막 step인 경우 종료
                    if float(tau_t)==0.0:
                        break
                    x=x_t
                    t+=1

    def train_bellman_ford(self,train_graph_list,hidden_dim=32,lr=0.01,epochs=10):
        optimizer=torch.optim.Adam(self.model.parameters(), lr=lr)
        prec_criterion=torch.nn.CrossEntropyLoss()
        dist_criterion=torch.nn.MSELoss()
        ter_criterion = torch.nn.BCEWithLogitsLoss()
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        self.model.train()

        for epoch in tqdm(range(epochs),desc="Training..."):
            for train_graph in train_graph_list:
                data=from_networkx(train_graph) # nx graph to pyg Data
                N=data.x.size(0)
                edge_index=data.edge_index.to(device)
                edge_attr=data.edge_attr.to(device)

                source_id=random.randint(0, N-1)

                h=torch.zeros((N,hidden_dim), dtype=torch.float32).detach().to(device) # h=(N,hidden_dim)
                last_p_label,last_x_label=Data_Processor.compute_shortest_path_and_predecessor(graph=train_graph,source_id=source_id)
                last_p_label=last_p_label.to(device)
                last_x_label=last_x_label.to(device)

                # initialize step
                graph_0,_,x_0=Data_Processor.compute_bellman_ford_step(graph=train_graph,source_id=source_id,init=True)
                x=x_0.detach().to(device) 
                graph_t=graph_0

                t=0
                while t < N:
                    optimizer.zero_grad()
                    graph_t,p_t,x_t=Data_Processor.compute_bellman_ford_step(graph=graph_t,source_id=source_id)
                    p_t=p_t.to(device)
                    x_t=x_t.to(device)

                    # get model output
                    output=self.model(x=x,edge_index=edge_index,edge_attr=edge_attr,pre_h=h)
                    h=output['h'].detach() # h=(N,hidden_dim)
                    prec=output['prec'] # prec=(N,1)
                    dist=output['dist'] # dist=(N,1)
                    tau=output['tau'] # tau=(1,1)

                    # set terminate standard
                    tau_t=self.compare_tensors(tensor1=last_x_label,tensor2=x_t).to(device) # 마지막 step이면 0.0 아니면 1.0

                    # 손실 함수 계산 및 오류 역전파 수행
                    precessor_loss=prec_criterion(prec,p_t.squeeze()) # prec=(N,N), p_t=(N,) 형태로 입력되어야 한다
                    distance_loss=dist_criterion(dist.squeeze(),x_t.squeeze())
                    terminate_loss=ter_criterion(tau.squeeze(),tau_t.squeeze())
                    total_loss=distance_loss+precessor_loss+terminate_loss
                    total_loss.backward()
                    optimizer.step()

                    # # 마지막 step인 경우 종료
                    if float(tau_t)==0.0:
                        break

                    x=x_t
                    t+=1
    
    def train_bellman_ford_distance(self,train_graph_list,hidden_dim=32,lr=0.01,epochs=10):
        optimizer=torch.optim.Adam(self.model.parameters(), lr=lr)
        dist_criterion=torch.nn.MSELoss()
        ter_criterion = torch.nn.BCEWithLogitsLoss()
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        self.model.train()

        for epoch in tqdm(range(epochs),desc="Training..."):
            for train_graph in train_graph_list:
                data=from_networkx(train_graph) # nx graph to pyg Data
                N=data.x.size(0)
                edge_index=data.edge_index.to(device)
                edge_attr=data.edge_attr.to(device)

                source_id=random.randint(0, N-1)

                h=torch.zeros((N,hidden_dim), dtype=torch.float32).detach().to(device) # h=(N,hidden_dim)
                _,last_x_label=Data_Processor.compute_shortest_path_and_predecessor(graph=train_graph,source_id=source_id)
                last_x_label=last_x_label.to(device)

                # initialize step
                graph_0,_,x_0=Data_Processor.compute_bellman_ford_step(graph=train_graph,source_id=source_id,init=True)
                x=x_0.detach().to(device) 
                graph_t=graph_0

                t=0
                while t < N:
                    optimizer.zero_grad()
                    graph_t,_,x_t=Data_Processor.compute_bellman_ford_step(graph=graph_t,source_id=source_id)
                    x_t=x_t.to(device)

                    # get model output
                    output=self.model(x=x,edge_index=edge_index,edge_attr=edge_attr,pre_h=h)
                    h=output['h'].detach() # h=(N,hidden_dim)
                    dist=output['dist'] # dist=(N,1)
                    tau=output['tau'] # tau=(1,1)

                    # set terminate standard
                    tau_t=self.compare_tensors(tensor1=last_x_label,tensor2=x_t).to(device) # 마지막 step이면 0.0 아니면 1.0

                    # 손실 함수 계산 및 오류 역전파 수행
                    distance_loss=dist_criterion(dist.squeeze(),x_t.squeeze())
                    terminate_loss=ter_criterion(tau.squeeze(),tau_t.squeeze())
                    total_loss=distance_loss+terminate_loss
                    total_loss.backward()
                    optimizer.step()

                    # # 마지막 step인 경우 종료
                    if float(tau_t)==0.0:
                        break

                    x=x_t
                    t+=1

    def evaluate_bfs(self,test_graph_list,model_file_name,hidden_dim=32):
        model=BFS_Neural_Execution(hidden_dim=hidden_dim)
        load_path=os.path.join(os.getcwd(), "inference",model_file_name+".pt")
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.load_state_dict(torch.load(load_path))
        model.eval()

        acc_dict_list=[]

        with torch.no_grad():
            for test_graph in tqdm(test_graph_list,desc="Evaluating..."):
                step_acc_list=[]

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
                graph_t=graph_0

                last_y=torch.zeros_like(x).to(device)
                t=0
                while t < N:
                    graph_t,x_t=Data_Processor.compute_bfs_step(graph=graph_t,source_id=source_id)
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
                    step_acc=self.compute_bfs_accuracy(y=cls_y,label=x_t)
                    step_acc_list.append(step_acc)

                    # set x to cls_y and last_y to cls_y
                    x=cls_y
                    last_y=cls_y

                    # terminate
                    tau=F.sigmoid(tau)
                    if tau.item()<=0.5:
                        break
                    t+=1
                
                # compute step acc and last acc
                step_acc_avg=np.mean(step_acc_list)
                last_acc=self.compute_bfs_accuracy(y=last_y,label=last_x_label)
                acc_dict={}
                acc_dict['step_acc_avg']=step_acc_avg
                acc_dict['last_acc']=last_acc
                acc_dict_list.append(acc_dict)

        k=0
        for acc_dict in acc_dict_list:
            k+=1
            print(f"{k} test graph step_acc_avg: {acc_dict['step_acc_avg']:.2%} and last_acc: {acc_dict['last_acc']:.2%}")
            print()

    def evaluate_bf(self,test_graph_list,model_file_name,hidden_dim=32):
        model=BF_Neural_Execution(hidden_dim=hidden_dim)
        load_path=os.path.join(os.getcwd(), "inference",model_file_name+".pt")
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.load_state_dict(torch.load(load_path))
        model.eval()

        acc_dict_list=[]

        with torch.no_grad():
            for test_graph in tqdm(test_graph_list,desc="Evaluating..."):
                step_prec_acc_list=[]
                step_dist_acc_list=[]
                data=from_networkx(test_graph) # nx graph to pyg Data
                N=data.x.size(0)
                edge_index=data.edge_index.to(device)
                edge_attr=data.edge_attr.to(device)
                source_id=random.randint(0, N-1)

                
                h=torch.zeros((N,hidden_dim), dtype=torch.float32).to(device) # h=(N,hidden_dim)
                last_p_label,last_x_label=Data_Processor.compute_shortest_path_and_predecessor(graph=test_graph,source_id=source_id)
                last_p_label=last_p_label.to(device)
                last_x_label=last_x_label.to(device)

                # initialize step
                graph_0,_,x_0=Data_Processor.compute_bellman_ford_step(graph=test_graph,source_id=source_id,init=True)
                x=x_0.to(device) # x=(N,1)
                graph_t=graph_0

                last_p=torch.zeros_like(x).to(device)
                last_x=torch.zeros_like(x).to(device)
                t=0
                while t < N:
                    graph_t,p_t,x_t=Data_Processor.compute_bellman_ford_step(graph=graph_t,source_id=source_id)
                    x_t=x_t.to(device)

                    # get model output
                    output=model(x=x,edge_index=edge_index,edge_attr=edge_attr,pre_h=h)
                    # get and set h
                    h=output['h'] # h=(N,hidden_dim)
                    # get predecessor,distance, tau
                    prec=output['prec'] # prec=(N,N)
                    dist=output['dist'] # dist=(N,1)
                    tau=output['tau'] # tau=(1,1)
                    
                    # compute step accuracy
                    step_prec_acc,step_dist_acc=self.compute_bf_accuracy(prec=prec,prec_label=p_t,dist=dist,dist_label=x_t)
                    step_prec_acc_list.append(step_prec_acc)
                    step_dist_acc_list.append(step_dist_acc)

                    # set x and last_x to dist
                    x=dist
                    last_p=prec
                    last_x=dist

                    # terminate
                    tau=F.sigmoid(tau)
                    if tau.item()<=0.5:
                        break
                    t+=1
                
                # compute step acc and last acc
                step_prec_acc_avg=np.mean(step_prec_acc_list)
                step_dist_acc_avg=np.mean(step_dist_acc_list)
                last_prec_acc,last_dist_acc=self.compute_bf_accuracy(prec=last_p,prec_label=last_p_label,dist=last_x,dist_label=last_x_label)
                acc_dict={}
                acc_dict['step_prec_acc_avg']=step_prec_acc_avg
                acc_dict['step_dist_acc_avg']=step_dist_acc_avg
                acc_dict['last_prec_acc']=last_prec_acc
                acc_dict['last_dist_acc']=last_dist_acc
                acc_dict_list.append(acc_dict)

        k=0
        for acc_dict in acc_dict_list:
            k+=1
            print(f"{k} test graph step_prec_acc_avg: {acc_dict['step_prec_acc_avg']:.2%} and last_prec_acc: {acc_dict['last_prec_acc']:.2%}")
            print(f"{k} test graph step_dist_acc_avg: {acc_dict['step_dist_acc_avg']:.2%} and last_dist_acc: {acc_dict['last_dist_acc']:.2%}")
            print()
    
    def evaluate_bf_distance(self,test_graph_list,model_file_name,hidden_dim=32):
        model=BF_Distance_Neural_Execution(hidden_dim=hidden_dim)
        load_path=os.path.join(os.getcwd(), "inference",model_file_name+".pt")
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.load_state_dict(torch.load(load_path))
        model.eval()

        acc_dict_list=[]

        with torch.no_grad():
            for test_graph in tqdm(test_graph_list,desc="Evaluating..."):
                step_dist_acc_list=[]
                data=from_networkx(test_graph) # nx graph to pyg Data
                N=data.x.size(0)
                edge_index=data.edge_index.to(device)
                edge_attr=data.edge_attr.to(device)
                source_id=random.randint(0, N-1)

                h=torch.zeros((N,hidden_dim), dtype=torch.float32).to(device) # h=(N,hidden_dim)
                _,last_x_label=Data_Processor.compute_shortest_path_and_predecessor(graph=test_graph,source_id=source_id)
                last_x_label=last_x_label.to(device)

                # initialize step
                graph_0,_,x_0=Data_Processor.compute_bellman_ford_step(graph=test_graph,source_id=source_id,init=True)
                x=x_0.to(device) # x=(N,1)
                graph_t=graph_0

                last_x=torch.zeros_like(x).to(device)
                t=0
                while t < N:
                    graph_t,_,x_t=Data_Processor.compute_bellman_ford_step(graph=graph_t,source_id=source_id)
                    x_t=x_t.to(device)

                    # get model output
                    output=model(x=x,edge_index=edge_index,edge_attr=edge_attr,pre_h=h)
                    # get and set h
                    h=output['h'] # h=(N,hidden_dim)
                    # get distance, tau
                    dist=output['dist'] # dist=(N,1)
                    tau=output['tau'] # tau=(1,1)
                    
                    # compute step accuracy
                    step_dist_acc=self.compute_bf_distance_accuracy(dist=last_x,dist_label=last_x_label)
                    step_dist_acc_list.append(step_dist_acc)

                    # set x and last_x to dist
                    x=dist
                    last_x=dist

                    # terminate
                    tau=F.sigmoid(tau)
                    if tau.item()<=0.5:
                        break
                    t+=1
                
                # compute step acc and last acc
                step_dist_acc_avg=np.mean(step_dist_acc_list)
                last_dist_acc=self.compute_bf_distance_accuracy(dist=last_x,dist_label=last_x_label)
                acc_dict={}
                acc_dict['step_dist_acc_avg']=step_dist_acc_avg
                acc_dict['last_dist_acc']=last_dist_acc
                acc_dict_list.append(acc_dict)

        k=0
        for acc_dict in acc_dict_list:
            k+=1
            print(f"{k} test graph step_dist_acc_avg: {acc_dict['step_dist_acc_avg']:.2%} and last_dist_acc: {acc_dict['last_dist_acc']:.2%}")
            print()