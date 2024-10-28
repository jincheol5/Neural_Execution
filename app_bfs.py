from utils import Data_Generator,Data_Loader,Data_Processor

dg=Data_Generator()
graph_list=dg.generate_graph_list(num_graph=1,num_node=5)

dl=Data_Loader()
loaded_graph_list=dl.load_pickle(file_name="train_graph_list")
graph=loaded_graph_list[0]
print(list(graph.edges))

dp=Data_Processor()
updated_graph=dp.compute_bfs_step(graph=graph,source_index=0,init=True)
for node_id,data in updated_graph.nodes(data=True):
    print("source node 0 to target node ",node_id," rechability: ",data['bfs'])

print()

updated_graph=dp.compute_bfs_step(graph=updated_graph,source_index=0,init=False)
for node_id,data in updated_graph.nodes(data=True):
    print("source node 0 to target node ",node_id," rechability: ",data['bfs'])