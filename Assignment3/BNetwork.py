from typing import List, Dict, Tuple
import json
from random import random
import itertools
import utils
import pprint

global_nodes_types_dict = {"vertex": {}, "edge_0": {}}  # type: Dict[str,Dict[str,Node]]
spontaneous_block_prob = 0.001

global_nodes = []
global_max_time = 0
global_number_of_iterations = 100000

global_edge_weights = {}
input_file = "graph1.json"


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
    
    def __str__(self):
        return pprint.pformat({'parents': self.parents,'table':self.table})

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

    return {key: (val / sum1) for key, val in W.items()} if sum1 else {}


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
    global global_nodes_types_dict, global_nodes, global_edge_weights
    with open(file_name) as f:
        json_dict = json.load(f)

    persistance = json_dict["persistance"]

    global_nodes = list(range(1, json_dict["nodes_num"] + 1))

    for vertex, prob in json_dict["node_prob"]:
        new_node = Node("vertex", str(vertex))
        new_node.table = {(): prob}
        new_node.initialized = True
        global_nodes_types_dict["vertex"][str(vertex)] = new_node

    global_edge_weights = {(n1,n2):weight for n1,n2,weight in json_dict["n1_n2_weight"]}
    for v1, v2, weight in json_dict["n1_n2_weight"]:
        new_id = "0_" + str(v1) + "_" + str(v2)
        new_node = Node("edge_0", new_id)
        global_nodes_types_dict["edge_0"][new_id] = new_node

        parent1 = global_nodes_types_dict["vertex"][str(v1)]
        parent2 = global_nodes_types_dict["vertex"][str(v2)]
        new_node.parents.extend([parent1, parent2])

        parent1.children.append(new_node)
        parent2.children.append(new_node)

        new_node.table.update({(1, 1): (1 - 0.16) / weight,
                               (1, 0): (1 - 0.4) / weight,
                               (0, 1): (1 - 0.4) / weight,
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


def add_evidence(evidence):
    res = utils.promptMenu("Which type of evidence do you want to add?", {"vertex": 0, "edge": 1})
    if not res:
        node = utils.promptIntegerFromRange("Please enter node number:",global_nodes)
        value = utils.promptMenu("Are there people in it", {"No": 0, "Yes": 1})
        id = str(node)
    else:
        node1 = utils.promptIntegerFromRange("Please enter node 1", global_nodes)
        node2 = utils.promptIntegerFromRange("Please enter node 2", global_nodes)
        time = utils.promptIntegerFromRange("Please enter time", list(range(0, global_max_time + 1)))
        value = utils.promptMenu("Is it blocked?", { "No": 0,"Yes": 1})
        id = str(time) + "_" + str(node1) + "_" + str(node2)

    evidence[id] = value


def reasoning(evidence):
    res = utils.promptMenu("Which type of reasoning do you want to do?", {"All vertices seperatly": 0,
                                                                          "All edges seperatly": 1,
                                                                          "Set of edges": 2,
                                                                          'Find best path between 2 nodes (bonus)':3})

    if res == 0:
        print('\nReasoning about query...\n')
        verteces = list(global_nodes_types_dict["vertex"].keys())
        resulting_join_distribution = likelihood_weightening(verteces,
                                                             evidence, global_nodes_types_dict,
                                                             global_number_of_iterations)
        if len(resulting_join_distribution) == 0:
            print('There seems to be a contradiction in given evidence!\nPlease reset evidence (or don\'t if you believe that there is just very low probability for this evidence) and try again!')
            return
        resulting_destributions = {}
        for i, vertex in zip(range(len(verteces)), verteces):
            summ = 0
            for values, join_prob in resulting_join_distribution.items():
                if values[i] == 1:
                    summ += join_prob
            resulting_destributions[vertex] = summ
        print('Result of query:\n')
        for v in resulting_destributions:
            print(f'\nVERTEX {v}:')
            print('Pr(Evacuees V%s) = %f'%(v,resulting_destributions[v]))
            print('Pr(~Evacuees V%s) = %f'%(v,1-resulting_destributions[v]))
            print('\n')
    elif res == 1:
        print('\nReasoning about query...\n')
        edges = [node for t, nodes in global_nodes_types_dict.items() if "edge" in t for node in nodes.keys()]
        resulting_join_distribution = likelihood_weightening(edges,
                                                        evidence, global_nodes_types_dict, global_number_of_iterations)
        if len(resulting_join_distribution) == 0:
            print('There seems to be a contradiction in given evidence!\nPlease reset evidence and try again!')            
            return
        resulting_destributions = {}

        for i, edge in zip(range(len(edges)), edges):
            summ = 0
            for values, join_prob in resulting_join_distribution.items():
                if values[i] == 1:
                    summ += join_prob
            resulting_destributions[edge] = summ
        print('Result of query:\n')
        for e in resulting_destributions:
            e_name = f'({",".join(e.split("_")[1:])})'
            t = e.split("_")[0]
            print('Edge%s at time-step=%s:'%(e_name,t))
            print('Pr(Blocakge E%s) = %f'%(e_name,resulting_destributions[e]))
            print('Pr(~Blocakge E%s) = %f'%(e_name,1-resulting_destributions[e]))
            print('\n')

    elif res == 2:
        node1 = 0
        edges = utils.promptPath(global_edge_weights.keys())
        time = utils.promptIntegerFromRange("Please enter time", list(range(0, global_max_time + 1)))
        print('\nReasoning about query...\n')
        edges = [f"{time}_{n1}_{n2}" for (n1,n2) in edges]
        resulting_join_distribution = likelihood_weightening(edges,
                                                                 evidence, global_nodes_types_dict,global_number_of_iterations)
        if len(resulting_join_distribution) == 0:
            print('There seems to be a contradiction in given evidence!\nPlease reset evidence and try again!')
            return
        all_not_blocked = tuple ([0]*(len(edges)))
        path_p = ','.join([f'~Blocakge E({",".join(e.split("_")[1:])})' for e in edges])
        print('Result of query:\n')
        print('Pr(%s) = %f'%(path_p,resulting_join_distribution[all_not_blocked]))
    elif res == 3:
        node1 = utils.promptIntegerFromRange("Please enter node 1", global_nodes)
        node2 = utils.promptIntegerFromRange("Please enter node 2", global_nodes)
        required_paths = utils.findAllSimplePaths(node1,node2,global_nodes,global_edge_weights)
        time = 1
        print('\nReasoning about query...\n')
        paths_considered = []
        for required_path in required_paths:
            edges = []
            key = '->'.join([str(n) for n in required_path])
            for i in range(len(required_path)-1):
                node1 = required_path[i]
                node2 = required_path[i+1]
                edges.append(f"{time}_"+str(min(node1,node2)) + "_" + str(max(node1,node2)))
            resulting_join_distribution = likelihood_weightening(edges,
                                                                    evidence, global_nodes_types_dict,global_number_of_iterations)
            if len(resulting_join_distribution) == 0:
                print('There seems to be a contradiction in given evidence!\nPlease reset evidence and try again!')
                return
            all_not_blocked = tuple ([0]*(len(required_path) - 1))
            path_p = ','.join([f'~Blocakge E({",".join(e.split("_")[1:])})' for e in edges])
            paths_considered.append((key,resulting_join_distribution[all_not_blocked],path_p))
        paths_considered = sorted(paths_considered,key=lambda x: -x[1])
        print('Result of query:\n')
        for i,(k,p,s) in enumerate(paths_considered):
            print(f'\n=====The considered path {k} is ranked ({i+1}/{len(paths_considered)})=====\n')
            print('Pr(%s) = %f'%(s,p))
            
    


def main_menu():
    q = False
    evidence = {}
    while not q:
        res = utils.promptMenu("\n\nChoose an action\n", {"Reset evidence": (lambda: evidence.clear()),
                                                    "Add evidence": (lambda: add_evidence(evidence)),
                                                    "Do reasoning": lambda: reasoning(evidence),
                                                    "Quit": lambda: True})
        q = res()


if __name__ == '__main__':
    global_max_time = 3
    load_network(input_file, global_max_time)
    main_menu()
