from utils import Data_Generator,Data_Loader

# class instance 생성
dg=Data_Generator()
dl=Data_Loader()

### train, test graph list 생성 및 저장
# train = 각 노드 수가 20개인 100개의 graph
# test = 각 노드 수가 20, 30, 50개인 graph 각 10개 => 30개 
train_graph_list=dg.generate_graph_list(num_graph=100,num_node=20)
test_graph_list=[]
test_graph_list.extend(dg.generate_graph_list(num_graph=10,num_node=20))
test_graph_list.extend(dg.generate_graph_list(num_graph=10,num_node=30))
test_graph_list.extend(dg.generate_graph_list(num_graph=10,num_node=50))
dl.save_pickle(data=train_graph_list,file_name="train_graph_list")
dl.save_pickle(data=test_graph_list,file_name="test_graph_list")
