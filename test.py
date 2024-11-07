from utils import Data_Loader

dl=Data_Loader()
test_graph_list=dl.load_pickle(file_name="test_graph_list")
print(len(test_graph_list))