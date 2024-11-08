import os
import random
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss
from torch_geometric.utils.convert import from_networkx
from utils import Data_Processor
from model import BFS_Neural_Execution



class Model_Trainer:
    def __init__(self,model=None):
        self.model=model
    
    def set_model(self,model):
        self.model=model
    
    def compute_accuracy(self,y,label):
        # y, label은 gpu 위에 있으나, 결과값 acc는 cpu로 반환
        num_nodes=y.size(0)
        correct=(y == label).sum().item()
        acc=correct/num_nodes
        return acc
    
    def compare_tensors(self,tensor1,tensor2):
        if torch.equal(tensor1, tensor2):
            # 값이 동일하면 (1, 1) 형태의 0.0 텐서 반환
            return torch.tensor([[0.0]])
        else:
            # 값이 다르면 (1, 1) 형태의 1.0 텐서 반환
            return torch.tensor([[1.0]])
    
    def save_model_state_dict(self,model_name):
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
                num_nodes=data.x.size(0)
                edge_index=data.edge_index.to(device)
                edge_attr=data.edge_attr.to(device)

                source_id=random.randint(0, num_nodes - 1)

                # initialize h
                h=torch.zeros((num_nodes,hidden_dim), dtype=torch.float32).detach().to(device) # h=(num_nodes,hidden_dim)

                # for termination
                last_x_label=Data_Processor.compute_reachability(graph=train_graph,source_id=source_id).to(device)

                # initialize step
                graph_0,x_0=Data_Processor.compute_bfs_step(graph=train_graph,source_id=source_id,init=True)
                x=x_0.detach().to(device) # x=(num_nodes,1)

                graph_t=graph_0

                t=1
                while t <= num_nodes:
                    optimizer.zero_grad()
                    graph_t,x_t=Data_Processor.compute_bfs_step(graph=graph_t,source_id=source_id)
                    x_t=x_t.to(device)

                    # get model output
                    output=self.model(x=x,edge_index=edge_index,edge_attr=edge_attr,pre_h=h)
                    h=output['h'].detach() # h=(num_nodes,hidden_dim)
                    y=output['y'] # y=(num_nodes,1)
                    cls_y=(y > 0.5).float() # y 값을 1.0 or 0.0으로 변환, GPU로 유지
                    ter=output['ter'] # ter=(1,1)

                    # set terminate standard
                    ter_std=self.compare_tensors(tensor1=last_x_label,tensor2=x_t).to(device) # 마지막 step이면 0.0 아니면 1.0

                    # 손실 함수 계산 및 오류 역전파 수행
                    class_loss=criterion(y,x_t)
                    terminate_loss=criterion(ter,ter_std)
                    total_loss=class_loss+terminate_loss
                    total_loss.backward()
                    optimizer.step()

                    # # 마지막 step인 경우 종료
                    if float(ter_std)==0.0:
                        break

                    # Noisy Teacher Forcing
                    # 학습 과정에서 모델이 자신의 예측값 대신 정답 (ground-truth)을 입력으로 받는 기법
                    # 50% 확률로 정답 힌트를 모델에 입력으로 주어, 학습이 안정되도록 돕는다
                    # 테스트 시에는 이전 단계에서 디코딩된 힌트를 그대로 다음 단계의 입력으로 사용한다
                    # set x, y값의 일부만 다음 time step의 x 값으로 전달
                    # next_input = torch.zeros_like(cls_y).to(device)   # (num_nodes, 1) 형태의 빈 텐서 GPU에서 생성
                    # for i in range(num_nodes):
                    #     # 각 노드에 대해 확률적으로 y 또는 x_t 선택
                    #     if random.random() < 0.5:
                    #         next_input[i] = x_t[i]  # 실제 라벨 값 선택
                    #     else:
                    #         next_input[i] = cls_y[i]  # 예측값 선택
                    # x=next_input.detach()
                    x=x_t
                    t+=1

    def test_bfs(self,test_graph_list,hidden_dim=32):
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        self.model.eval()

        acc_dict_list=[]

        with torch.no_grad():
            for test_graph in tqdm(test_graph_list,desc="Evaluating..."):
                step_acc_list=[]

                data=from_networkx(test_graph) # nx graph to pyg Data
                num_nodes=data.x.size(0)
                edge_index=data.edge_index.to(device)
                edge_attr=data.edge_attr.to(device)

                source_id=random.randint(0, num_nodes - 1)

                # initialize h
                h=torch.zeros((num_nodes,hidden_dim), dtype=torch.float32).to(device) # h=(num_nodes,hidden_dim)

                # for termination
                last_x_label=Data_Processor.compute_reachability(graph=test_graph,source_id=source_id)
                last_x_label=last_x_label.to(device)

                # initialize step
                graph_0,x_0=Data_Processor.compute_bfs_step(graph=test_graph,source_id=source_id,init=True)
                x=x_0.to(device) # x=(num_nodes,1)
                graph_t=graph_0

                last_y=torch.zeros_like(x).to(device)
                t=1
                while t <= num_nodes:
                    graph_t,x_t=Data_Processor.compute_bfs_step(graph=graph_t,source_id=source_id)
                    x_t=x_t.to(device)

                    # get model output
                    output=self.model(x=x,edge_index=edge_index,edge_attr=edge_attr,pre_h=h)
                    # get and set h
                    h=output['h'] # h=(num_nodes,hidden_dim)
                    # get y, ter
                    y=output['y'] # y=(num_nodes,1)
                    cls_y=(y > 0.5).float() # y 값을 1.0 or 0.0으로 변환, GPU로 유지
                    ter=output['ter'] # ter=(1,1)
                    
                    # compute step accuracy
                    step_acc=self.compute_accuracy(y=cls_y,label=x_t)
                    step_acc_list.append(step_acc)

                    # set x to cls_y and last_y to cls_y
                    x=cls_y
                    last_y=cls_y

                    # terminate
                    tau=F.sigmoid(ter)
                    if tau.item()<=0.5:
                        break
                    t+=1
                
                # compute step acc and last acc
                step_acc_avg=np.mean(step_acc_list)
                last_acc=self.compute_accuracy(y=last_y,label=last_x_label)
                acc_dict={}
                acc_dict['step_acc_avg']=step_acc_avg
                acc_dict['last_acc']=last_acc
                acc_dict_list.append(acc_dict)

        k=0
        for acc_dict in acc_dict_list:
            k+=1
            print(f"{k} test graph step_acc_avg: {acc_dict['step_acc_avg']:.2%} and last_acc: {acc_dict['last_acc']:.2%}")
            print()


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
                num_nodes=data.x.size(0)
                edge_index=data.edge_index.to(device)
                edge_attr=data.edge_attr.to(device)

                source_id=random.randint(0, num_nodes - 1)

                # initialize h
                h=torch.zeros((num_nodes,hidden_dim), dtype=torch.float32).to(device) # h=(num_nodes,hidden_dim)

                # for termination
                last_x_label=Data_Processor.compute_reachability(graph=test_graph,source_id=source_id)
                last_x_label=last_x_label.to(device)

                # initialize step
                graph_0,x_0=Data_Processor.compute_bfs_step(graph=test_graph,source_id=source_id,init=True)
                x=x_0.to(device) # x=(num_nodes,1)
                graph_t=graph_0

                last_y=torch.zeros_like(x).to(device)
                t=1
                while t <= num_nodes:
                    graph_t,x_t=Data_Processor.compute_bfs_step(graph=graph_t,source_id=source_id)
                    x_t=x_t.to(device)

                    # get model output
                    output=model(x=x,edge_index=edge_index,edge_attr=edge_attr,pre_h=h)
                    # get and set h
                    h=output['h'] # h=(num_nodes,hidden_dim)
                    # get y, ter
                    y=output['y'] # y=(num_nodes,1)
                    cls_y=(y > 0.5).float() # y 값을 1.0 or 0.0으로 변환, GPU로 유지
                    ter=output['ter'] # ter=(1,1)
                    
                    # compute step accuracy
                    step_acc=self.compute_accuracy(y=cls_y,label=x_t)
                    step_acc_list.append(step_acc)

                    # set x to cls_y and last_y to cls_y
                    x=cls_y
                    last_y=cls_y

                    # terminate
                    tau=F.sigmoid(ter)
                    if tau.item()<=0.5:
                        break
                    t+=1
                
                # compute step acc and last acc
                step_acc_avg=np.mean(step_acc_list)
                last_acc=self.compute_accuracy(y=last_y,label=last_x_label)
                acc_dict={}
                acc_dict['step_acc_avg']=step_acc_avg
                acc_dict['last_acc']=last_acc
                acc_dict_list.append(acc_dict)

        k=0
        for acc_dict in acc_dict_list:
            k+=1
            print(f"{k} test graph step_acc_avg: {acc_dict['step_acc_avg']:.2%} and last_acc: {acc_dict['last_acc']:.2%}")
            print()

    def evaluate_bfs_dataset(self,test_graph,model_file_name,src_list,hidden_dim=32):
        model=BFS_Neural_Execution(hidden_dim=32)
        load_path=os.path.join(os.getcwd(), "inference",model_file_name+".pt")
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.load_state_dict(torch.load(load_path))
        model.eval()

        print("node :",test_graph.number_of_nodes())
        print("edge: ",test_graph.number_of_edges())

        with torch.no_grad():
            step_acc_list=[]

            data=from_networkx(test_graph) # nx graph to pyg Data
            num_nodes=data.x.size(0)
            edge_index=data.edge_index.to(device)
            edge_attr=data.edge_attr.to(device)

            for source_id in src_list:
                # initialize h
                h=torch.zeros((num_nodes,hidden_dim), dtype=torch.float32).to(device) # h=(num_nodes,hidden_dim)

                # for termination
                last_x_label=Data_Processor.compute_reachability(graph=test_graph,source_id=source_id)
                last_x_label=last_x_label.to(device)

                # initialize step
                graph_0,x_0=Data_Processor.compute_bfs_step(graph=test_graph,source_id=source_id,init=True)
                x=x_0.to(device) # x=(num_nodes,1)
                graph_t=graph_0

                last_y=torch.zeros_like(x).to(device)
                t=1
                while t <= num_nodes:
                    graph_t,x_t=Data_Processor.compute_bfs_step(graph=graph_t,source_id=source_id)
                    x_t=x_t.to(device)

                    # get model output
                    output=model(x=x,edge_index=edge_index,edge_attr=edge_attr,pre_h=h)
                    # get and set h
                    h=output['h'] # h=(num_nodes,hidden_dim)
                    # get y, ter
                    y=output['y'] # y=(num_nodes,1)
                    cls_y=(y > 0.5).float() # y 값을 1.0 or 0.0으로 변환, GPU로 유지
                    ter=output['ter'] # ter=(1,1)
                    
                    # compute step accuracy
                    step_acc=self.compute_accuracy(y=cls_y,label=x_t)
                    step_acc_list.append(step_acc)

                    # set x to cls_y and last_y to cls_y
                    x=cls_y
                    last_y=cls_y

                    # terminate
                    tau=F.sigmoid(ter)
                    if tau.item()<=0.5:
                        break
                    t+=1

                # compute step acc and last acc
                step_acc_avg=np.mean(step_acc_list)
                last_acc=self.compute_accuracy(y=last_y,label=last_x_label)

                print(f"test graph step_acc_avg: {step_acc_avg:.2%} and last_acc: {last_acc:.2%}")


    def test_bfs_step(self,test_graph,model_file_name,hidden_dim=32):
        model=BFS_Neural_Execution(hidden_dim=32)
        load_path=os.path.join(os.getcwd(), "inference",model_file_name+".pt")
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.load_state_dict(torch.load(load_path))
        model.eval()

        with torch.no_grad():
            step_acc_list=[]

            data=from_networkx(test_graph) # nx graph to pyg Data
            num_nodes=data.x.size(0)
            edge_index=data.edge_index.to(device)
            edge_attr=data.edge_attr.to(device)

            source_id=random.randint(0, num_nodes - 1)

            # initialize h
            h=torch.zeros((num_nodes,hidden_dim), dtype=torch.float32).to(device) # h=(num_nodes,hidden_dim)

            # for termination
            last_x_label=Data_Processor.compute_reachability(graph=test_graph,source_id=source_id)
            last_x_label=last_x_label.to(device)

            # initialize step
            graph_0,x_0=Data_Processor.compute_bfs_step(graph=test_graph,source_id=source_id,init=True)
            x=x_0.to(device) # x=(num_nodes,1)
            graph_t=graph_0

            last_y=torch.zeros_like(x).to(device)
            t=1
            while t <= num_nodes:
                graph_t,x_t=Data_Processor.compute_bfs_step(graph=graph_t,source_id=source_id)
                x_t=x_t.to(device)

                # get model output
                output=model(x=x,edge_index=edge_index,edge_attr=edge_attr,pre_h=h)
                # get and set h
                h=output['h'] # h=(num_nodes,hidden_dim)
                # get y, ter
                y=output['y'] # y=(num_nodes,1)
                cls_y=(y > 0.5).float() # y 값을 1.0 or 0.0으로 변환, GPU로 유지
                ter=output['ter'] # ter=(1,1)
                
                # compute step accuracy
                step_acc=self.compute_accuracy(y=cls_y,label=x_t)
                step_acc_list.append(step_acc)

                # set x to cls_y and last_y to cls_y
                x=cls_y
                last_y=cls_y

                # print step prediction
                print(f"{t} step prediction: {cls_y.cpu()}")
                print()

                # terminate
                tau=F.sigmoid(ter)
                if tau.item()<=0.5:
                    break
                t+=1

            # compute step acc and last acc
            step_acc_avg=np.mean(step_acc_list)
            last_acc=self.compute_accuracy(y=last_y,label=last_x_label)

            print(f"test graph step_acc_avg: {step_acc_avg:.2%} and last_acc: {last_acc:.2%}")

