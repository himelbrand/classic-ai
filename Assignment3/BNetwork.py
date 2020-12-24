from typing import List, Dict, Tuple
import json
from random import random
import itertools

global_nodes_types_dict = {"vertex": {}, "edge_0": {}}  # type: Dict[str,Dict[str,Node]]
spontaneous_block_prob = 0.001

input_file = "graph2.json"


class Node:
    def __init__(self, type, id: str, ):
        self.id = id  # type: str
        self.type = type
        self.parents = []  # type: List[Node]
        self.children = []  # type: List[Node]
        self.table = {}  # type: Dict[Tuple,float]
        self.initialized = False

    def get_parents_ids(self):
        return [parent.id for parent in self.parents]

    def get_probability(self, self_value, parents_values):
        return abs(self_value - (1 - self.table[tuple(parents_values)]))

    def get_sample(self, parents_values):
        probability = self.table[tuple(parents_values)]
        if random() <= probability:
            return 1
        return 0

    def get_domain(self):
        return (1, 0)


def likelihood_weightening(x_ids, evidence_dict, network, N):
    nodes = {node_id: node for nodes in network.values() for node_id, node in nodes.items()}
    x_nodes_domains = [nodes[x_id].get_domain() for x_id in x_ids]
    x_domain = itertools.product(*x_nodes_domains)
    W = {tuple(domain): 0 for domain in x_domain}

    for _ in range(N):
        network_sample, w = weighted_sample(network, evidence_dict)
        x = tuple([network_sample[id] for id in x_ids])
        W[x] += w

    sum1 = sum(W.values())

    return {key: (val / sum1) for key, val in W.items()}


def weighted_sample(network, evidence_dict):
    evidence_dict = dict(evidence_dict)
    topological_nodes = [node for nodes in network.values() for node in nodes.values()]
    w = 1

    for node in topological_nodes:
        parents = node.get_parents_ids()
        parents_values = tuple([evidence_dict[parent] for parent in parents])
        if node.id in evidence_dict.keys():
            node_value = evidence_dict[node.id]
            w *= node.get_probability(node_value, parents_values)
        else:
            sample = node.get_sample(parents_values)
            evidence_dict[node.id] = sample

    return evidence_dict, w


def load_network(file_name, time_steps: int):
    global global_nodes_types_dict
    with open(file_name) as f:
        json_dict = json.load(f)

    persistance = json_dict["persistance"]

    nodes = list(range(1, json_dict["nodes_num"] + 1))

    for vertex, prob in json_dict["node_prob"]:
        new_node = Node("vertex", str(vertex))
        new_node.table = {(): prob}
        new_node.initialized = True
        global_nodes_types_dict["vertex"][str(vertex)] = new_node

    for v1, v2, weight in json_dict["n1_n2_weight"]:
        new_id = "0_" + str(v1) + "_" + str(v2)
        new_node = Node("edge_0", new_id)
        global_nodes_types_dict["edge_0"][new_id] = new_node

        parent1 = global_nodes_types_dict["vertex"][str(v1)]
        parent2 = global_nodes_types_dict["vertex"][str(v2)]
        new_node.parents.extend([parent1, parent2])

        parent1.children.append(new_node)
        parent2.children.append(new_node)

        new_node.table.update({(1, 1): 1 - 0.16 / (weight ** 2),
                               (1, 0): 1 - 0.4 / weight,
                               (0, 1): 1 - 0.4 / weight,
                               (0, 0): spontaneous_block_prob})
        new_node.initialized = True

    for time in range(1, time_steps + 1):

        new_type = "edge_" + str(time)
        parent_type = "edge_" + str(time - 1)
        global_nodes_types_dict[new_type] = {}
        parents = global_nodes_types_dict[parent_type].values()

        for parent in parents:
            new_id = str(time) + parent.id[parent.id.find("_"):]
            new_node = Node(new_type, new_id)

            new_node.parents = [parent]
            parent.children = [new_node]

            new_node.table = {(1,): persistance, (0,): spontaneous_block_prob}
            new_node.initialized = True

            global_nodes_types_dict[new_type][new_id] = new_node

    print("Stop")


if __name__ == '__main__':
    load_network(input_file, 1)
    likelihood_weightening(["0_1_2"],{"1":1,"2":1,"1_1_2":0},global_nodes_types_dict,10000)