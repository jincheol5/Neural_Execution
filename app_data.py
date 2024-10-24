from utils import DataGenerator,DataLoader

dg=DataGenerator()
graph_list=dg.generate_graph_list(num_graph=5,num_node=10)

dl=DataLoader()
dl.save_data(data=graph_list,file_name="train_graph_list")
loaded_graph_list=dl.load_data(file_name="train_graph_list")

print(loaded_graph_list)