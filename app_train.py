from utils import Data_Loader
from model import BFS_Neural_Execution
from model_train import Model_Trainer

dl=Data_Loader()
model=BFS_Neural_Execution(hidden_dim=32)
model_trainer=Model_Trainer(model=model)

train_graph_list=dl.load_pickle(file_name="train_graph_list")
test_graph_list=dl.load_pickle(file_name="test_graph_list")

model_trainer.train_bfs(train_graph_list=train_graph_list,hidden_dim=32)
model_trainer.test_bfs(test_graph_list=test_graph_list,hidden_dim=32)