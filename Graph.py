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

    def min_spanning_tree_kruskal(self,blocked_edges):
        graph = self.graph
        weights = self.weights

        F = []
        nodes = list(graph.keys())

        sets = [{node} for node in nodes]

        sorted_edges_and_weights = [(edge_and_w[0],edge_and_w[1]) for edge_and_w in sorted(weights.items(),key=lambda x : x[1])]

        tree_edges = []
        tree_weight = 0
        for edge,weight in sorted_edges_and_weights:
            if edge in blocked_edges:
                continue
            node1, node2 = edge
            set1 = self.find_set(node1,sets)
            if node2 in set1:
                continue
            set2 = self.find_set(node2,sets)
            sets.remove(set1)
            sets.remove(set2)
            sets.append(set1 | set2)
            tree_edges.append((node1,node2))
            tree_weight += weight
            if len(sets) == 1:
                break

        if len(sets) > 1:
            tree_edges = None
            tree_weight = float("inf")

        return tree_edges,tree_weight

    def find_set(self,item,set_list):
        for set1 in set_list:
            if item in set1:
                return set1
        return None