from utils import Data_Loader,Data_Generator
from model_train import Model_Trainer

graph=Data_Generator().generate_test_graph()

mt=Model_Trainer()
mt.test_bfs_one(test_graph=graph,model_file_name="neural_execution_bfs",hidden_dim=32)