from typing import Dict , List, Set, Tuple
import json
import random
import heapq



class Graph:

    def __init__(self,graph,weights):
        self.graph = graph            #type: Dict[int,List[Tuple[int,float]]]
        self.weights = weights        #type: Dict[Tuple[int,int],int]

    def get_neigbours_and_weights (self,node ):
        neighbors =  self.graph[node]
        neigbours_and_weights = []

        for neighb in neighbors:
            neigbours_and_weights.append((neighb,self.get_weight(node,neighb)))

        return neigbours_and_weights

    def is_neighbours(self,n1,n2):
        neigbours_and_weights = self.get_neigbours_and_weights(n1)

        neigbours, _ = zip(*neigbours_and_weights)
        return n2 in neigbours

    def get_weight(self,node1,node2):
        return  self.weights[(min(node1,node2),max(node1,node2))]

    def get_shortest_path_Dijk(self, node1, node2, blocked=None):
        if blocked is None:
            blocked = []

        nodes = self.graph.keys()
        node_parents = { node:None for node in nodes}
        node_dists = { node:[float("inf"),node] for node in nodes}

        node_dists[node1] = [0,node1]
        Q = [[0,node1]]


        while len(Q) > 0:
            heapq.heapify(Q)
            dist,node = heapq.heappop(Q)

            if node == node2:break

            neighbs_weights = self.get_neigbours_and_weights(node)

            print(neighbs_weights)
            for neighb,weight in neighbs_weights:
                if (node,neighb) in blocked:
                    continue
                if node_dists[neighb][0] > dist + weight :
                    node_dists[neighb][0] = dist + weight
                    node_parents[neighb] = node
                    Q.append(node_dists[neighb])

        if node_dists[node2][0] == float("inf"):
            return float("inf"),[]

        temp_node = node2
        path = []
        while True:
            path.append(temp_node)
            if temp_node == node1:
                break
            temp_node = node_parents[temp_node]
        path.reverse()
        return node_dists[node2][0],path



