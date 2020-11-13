from typing import Dict, List, Set, Tuple
import json
import random
import heapq


class Graph:

    def __init__(self, graph, weights):
        self.graph = graph  # type: Dict[int,List[int]]         # Graph : key - node ,
        # value - list of it's neighbors

        self.weights = weights  # type: Dict[Tuple[int,int],int]    # List of blocked edges ( for each edge two
        # tuples (node1,node2 ) and (node2,node1) )

    def get_neigbours_and_weights(self, node):
        """For node , return the list of tuples ( neigbhor , edge_weight ) """
        neighbors = self.graph[node]
        neigbours_and_weights = []

        for neighb in neighbors:
            neigbours_and_weights.append((neighb, self.get_weight(node, neighb)))

        return neigbours_and_weights

    def is_neighbours(self, n1, n2):
        # Check whether two nodes are neighbors
        neigbours_and_weights = self.get_neigbours_and_weights(n1)

        neigbours, _ = zip(*neigbours_and_weights)
        return n2 in neigbours

    def get_weight(self, node1, node2):
        # Get weight of the edge between two nodes and infinity if there is no edge
        return self.weights.get((min(node1, node2), max(node1, node2)), float("inf"))

    def get_shortest_path_Dijk(self, node1, node2, blocked=None,nodes_to_avoid = None):

        """Compute shortest path between two nodes ( Dijkstra ) without using the edges in the blocked list.
        Return value : The weight of the shortest path and the list of nodes - the path itself , or infinity and empty list
         if there is no such a path"""
        if blocked is None:
            blocked = []

        if nodes_to_avoid is None:
            nodes_to_avoid = []
        nodes = self.graph.keys()
        node_parents = {node: None for node in nodes}
        node_dists = {node: [float("inf"), node] for node in nodes}

        node_dists[node1] = [0, node1]
        Q = [[0, node1]]

        while len(Q) > 0:
            heapq.heapify(Q)
            dist, node = heapq.heappop(Q)

            if node == node2: break

            neighbs_weights = self.get_neigbours_and_weights(node)


            for neighb, weight in neighbs_weights:
                if (node, neighb) in blocked or neighb in nodes_to_avoid:
                    continue

                if node_dists[neighb][0] > dist + weight:
                    node_dists[neighb][0] = dist + weight
                    node_parents[neighb] = node
                    Q.append(node_dists[neighb])

        if node_dists[node2][0] == float("inf"):
            return float("inf"), []

        temp_node = node2
        path = []
        while True:
            path.append(temp_node)
            if temp_node == node1:
                break
            temp_node = node_parents[temp_node]
        path.reverse()

        return node_dists[node2][0], path
