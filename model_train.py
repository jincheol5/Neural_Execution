import os
import random
from tqdm import tqdm
import numpy as np
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
            # 값이 다르면 (1, 1) 형태의 0.0 텐서 반환
            return torch.tensor([[1.0]])
    
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
                edge_index=data.edge_index
                edge_attr=data.edge_attr
                edge_index=edge_index.to(device)
                edge_attr=edge_attr.to(device)

                assert data.x is not None, "data.x가 None입니다. 그래프에 노드 특성이 없는지 확인하세요."
                assert edge_index.device == edge_attr.device, "edge_index와 edge_attr는 동일한 장치에 있어야 합니다."


                source_id=random.randint(0, num_nodes - 1)

                # initialize h
                h=torch.zeros((num_nodes,hidden_dim), dtype=torch.float32) # h=(num_nodes,hidden_dim)
                h=h.detach() # 값만 전달하기 위해
                h=h.to(device)

                # for termination
                last_x_label=self.dp.compute_reachability(graph=train_graph,source_id=source_id)
                last_x_label=last_x_label.to(device)

                # initialize step
                step_graph,x_0=self.dp.compute_bfs_step(graph=train_graph,source_id=source_id,init=True)
                x=x_0.to(device) # x=(num_nodes,1)

                t=0
                while t < num_nodes:
                    step_graph,step_x_label=self.dp.compute_bfs_step(graph=step_graph,source_id=source_id)
                    step_x_label=step_x_label.to(device)
                    optimizer.zero_grad()

                    # get model output
                    output=self.model(x=x,edge_index=edge_index,edge_attr=edge_attr,pre_h=h)
                    # get and set h
                    h=output['h'] # h=(num_nodes,hidden_dim)
                    # get y, ter
                    y=output['y'] # y=(num_nodes,1)
                    cls_y=(y > 0.5).float().to(device)  # y 값을 1.0 or 0.0으로 변환, GPU로 유지
                    cls_y=cls_y.detach()
                    ter=output['ter'] # ter=(1,1)
                    # set terminate standard
                    ter_std=self.compare_tensors(tensor1=last_x_label,tensor2=step_x_label).to(device) # 마지막 step이면 0.0 아니면 1.0

                    # 손실 함수 계산 및 오류 역전파 수행
                    class_loss=criterion(y,step_x_label)
                    terminate_loss=criterion(ter,ter_std)
                    total_loss=class_loss+terminate_loss
                    total_loss.backward()
                    optimizer.step()

                    # 마지막 step인 경우 종료
                    if float(ter_std)==0.0:
                        break

                    # Noisy Teacher Forcing
                    # 학습 과정에서 모델이 자신의 예측값 대신 정답 (ground-truth)을 입력으로 받는 기법
                    # 50% 확률로 정답 힌트를 모델에 입력으로 주어, 학습이 안정되도록 돕는다
                    # 테스트 시에는 이전 단계에서 디코딩된 힌트를 그대로 다음 단계의 입력으로 사용한다
                    # set x, y값의 일부만 다음 time step의 x 값으로 전달
                    next_input = torch.zeros_like(cls_y).to(device)  # (num_nodes, 1) 형태의 빈 텐서 GPU에서 생성
                    for i in range(num_nodes):
                        # 각 노드에 대해 확률적으로 y 또는 x_t 선택
                        if random.random() < 0.5:
                            next_input[i] = step_x_label[i]  # 실제 라벨 값 선택
                        else:
                            next_input[i] = cls_y[i]  # 예측값 선택
                    x=next_input
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
                edge_index=data.edge_index
                edge_attr=data.edge_attr
                edge_index=edge_index.to(device)
                edge_attr=edge_attr.to(device)

                source_id=random.randint(0, num_nodes - 1)

                # initialize h
                h=torch.zeros((num_nodes,hidden_dim), dtype=torch.float32) # h=(num_nodes,hidden_dim)
                h=h.to(device)

                # for termination
                last_x_label=self.dp.compute_reachability(graph=test_graph,source_id=source_id)
                last_x_label=last_x_label.to(device)

                # initialize step
                step_graph,x_0=self.dp.compute_bfs_step(graph=test_graph,source_id=source_id,init=True)
                x=x_0.to(device) # x=(num_nodes,1)

                last_y=torch.zeros_like(x, device=device)
                t=0
                while t < num_nodes:
                    step_graph,step_x_label=self.dp.compute_bfs_step(graph=step_graph,source_id=source_id)
                    step_x_label=step_x_label.to(device)

                    # get model output
                    output=self.model(x=x,edge_index=edge_index,edge_attr=edge_attr,pre_h=h)
                    # get and set h
                    h=output['h'] # h=(num_nodes,hidden_dim)
                    # get y, ter
                    y=output['y'] # y=(num_nodes,1)
                    cls_y=(y > 0.5).float().to(device)  # y 값을 1.0 or 0.0으로 변환, GPU로 유지
                    ter=output['ter'] # ter=(1,1)
                    
                    # compute step accuracy
                    step_acc=self.compute_accuracy(y=cls_y,label=step_x_label)
                    step_acc_list.append(step_acc)

                    # set x to cls_y and last_y to cls_y
                    x=cls_y
                    last_y=cls_y

                    # terminate
                    tau=torch.nn.Sigmoid(ter)
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
            print(k," test graph average step acc: ",acc_dict['step_acc_avg']," and last acc: ",acc_dict['last_acc'])
            print()