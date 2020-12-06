import heapq

def get_shortest_path_Dijk(G,origin,dest,blocked=None,nodes_to_avoid=None):

    """Compute shortest path between two nodes ( Dijkstra ) without using the edges in the blocked list.
    Return value : The weight of the shortest path and the list of nodes - the path itself , or infinity and empty list
     if there is no such a path"""
    
    if blocked is None:
        blocked = []

    if nodes_to_avoid is None:
        nodes_to_avoid = []
    nodes = G.graph.keys()
    node_parents = {node: None for node in nodes}
    node_dists = {node: [float("inf"), node] for node in nodes}

    node_dists[origin] = [0, origin]
    Q = [[0, origin]]

    while len(Q) > 0:
        heapq.heapify(Q)
        dist, node = heapq.heappop(Q)

        if node == dest: break

        neighbs_weights = G.get_neigbours_and_weights(node)


        for neighb, weight in neighbs_weights:
            if (node, neighb) in blocked or neighb in nodes_to_avoid:
                continue

            if node_dists[neighb][0] > dist + weight:
                node_dists[neighb][0] = dist + weight
                node_parents[neighb] = node
                Q.append(node_dists[neighb])

    if node_dists[dest][0] == float("inf"):
        return float("inf"), []

    temp_node = dest
    path = []
    while True:
        path.append(temp_node)
        if temp_node == origin: break
        temp_node = node_parents[temp_node]
    path.reverse()

    return node_dists[dest][0], path

def min_spanning_tree_kruskal(G,blocked_edges):
    def find_set(item,set_list):
        for set1 in set_list:
            if item in set1:
                return set1
        return None

    graph = G.graph
    weights = G.weights
    nodes = list(graph.keys())

    sets = [{node} for node in nodes]

    sorted_edges_and_weights = [(edge_and_w[0],edge_and_w[1]) for edge_and_w in sorted(weights.items(),key=lambda x : x[1])]

    tree_edges = []
    tree_weight = 0
    for edge,weight in sorted_edges_and_weights:
        if edge in blocked_edges:
            continue
        node1, node2 = edge
        set1 = find_set(node1,sets)
        
        if node2 in set1:
            continue

        set2 = find_set(node2,sets)
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

    