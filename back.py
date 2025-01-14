# N=graph.number_of_nodes()
#         if init:
#             for idx in range(N):
#                 graph.nodes[idx]['x']=0.0
#             graph.nodes[source_id]['x']=1.0
#         else:
#             graph_t=copy.deepcopy(graph) # 순회 내에서 업데이트 결과가 이후 결과에 영향을 미치지 않도록 복사 -> edge 순서 관계없이 각 단계 결과값 예측 가능
#             for idx in range(N):
#                 if graph.nodes[idx]['x']==1.0:
#                     for neighbor in graph.neighbors(idx):
#                         if graph.nodes[neighbor]['x']==0.0:
#                             graph_t.nodes[neighbor]['x']=1.0


# @staticmethod
#     def compute_reachability(graph,source_id):
#         nodes=list(graph.nodes())
#         result_tensor=torch.zeros((len(nodes),), dtype=torch.float32) # result_tensor=(N,1)

#         for tar in nodes:
#             if nx.has_path(graph,source=source_id,target=tar):
#                 result_tensor[tar]=1.0

#         return result_tensor

# @staticmethod
#     def compute_bellman_ford_step(graph,source_id=0,init=False):
#         N=graph.number_of_nodes()
#         x_t=torch.zeros((N,),dtype=torch.float32)
#         p_t=torch.zeros((N,),dtype=torch.int)
#         for idx in range(N):
#             x_t[idx]=graph.nodes[idx]['x']
#             p_t[idx]=graph.nodes[idx]['p']

#         if init:
#             # compute longest distance
#             copy_graph=copy.deepcopy(graph)
#             _, distance_dic=nx.bellman_ford_predecessor_and_distance(G=copy_graph, source=source_id, weight='w')
#             longest_shortest_path_distance=max(distance_dic.values())
#             longest_shortest_path_distance+=1

#             # initialize 
#             for idx in range(N):
#                 graph.nodes[idx]['x']=longest_shortest_path_distance
#                 x_t[idx]=longest_shortest_path_distance
#                 graph.nodes[idx]['p']=idx
#                 p_t[idx]=graph.nodes[idx]['p']=idx
#             graph.nodes[source_id]['x']=0.0
#             x_t[source_id]=0.0

#             return graph,p_t,x_t
#         else:
#             graph_t=copy.deepcopy(graph) # edge 순서에 영향 받지 않기 위해 복사 -> edge 순서 상관없이 각 단계 결과값 예측 가능
#             for src,tar in list(graph.edges()): # edge=(src,tar)
#                 if graph.nodes[src]['x']+graph.edges[(src, tar)]['w']<graph.nodes[tar]['x']:
#                     graph_t.nodes[tar]['x']=graph.nodes[src]['x']+graph.edges[(src, tar)]['w']
#                     x_t[tar]=graph.nodes[src]['x']+graph.edges[(src, tar)]['w']
#                     graph_t.nodes[tar]['p']=src
#                     p_t[tar]=src

#             return graph_t,p_t,x_t


# @staticmethod
#     def compute_shortest_path_and_predecessor(graph,source_id):
#         nodes=list(graph.nodes())
#         predecessor_tensor=torch.zeros((len(nodes),), dtype=torch.int) # predecessor_tensor=(N,1)
#         distance_tensor=torch.zeros((len(nodes),), dtype=torch.float32) # distance_tensor=(N,1)

#         predecessor_dic, distance_dic = nx.bellman_ford_predecessor_and_distance(G=graph, source=source_id, weight='w') # 연결되지 않은 노드=key도 없음, source 노드=key는 있지만 value=[]
#         predecessor_dic[source_id]=[source_id]

#         for tar in nodes:
#             if tar in predecessor_dic:
#                 predecessor_tensor[tar]=predecessor_dic[tar][0]
#                 distance_tensor[tar]=distance_dic[tar]
#             else:
#                 predecessor_tensor[tar]=tar
#         return predecessor_tensor,distance_tensor