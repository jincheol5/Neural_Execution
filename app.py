import networkx as nx
import numpy as np
from utils import Data_Generator,Data_Processor

# 가중 무방향 그래프 생성
G = nx.Graph()

# 엣지 추가 (엣지 속성 'weight' 사용)
G.add_edge(1, 2, edge_attr=0.1)
G.add_edge(2, 3, edge_attr=0.2)
G.add_edge(3, 4, edge_attr=0.1)
G.add_edge(1, 4, edge_attr=0.5)

# 다익스트라 알고리즘으로 predecessor와 최단 거리 계산
predecessor, distance = nx.bellman_ford_predecessor_and_distance(G, source=1, weight='edge_attr')

# 결과 출력
print("Predecessor:", predecessor)
print("Distance:", distance)
