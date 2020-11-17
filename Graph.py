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