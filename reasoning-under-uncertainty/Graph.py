from typing import List,Dict,Tuple
import json


def load_network(file_name):
    with open(file_name) as f:
        json_dict = json.load(f)

    persistance = json_dict["persistance"]

    nodes = list(range(1, json_dict["nodes_num"] + 1))

    node_prob = {node_prob[0]: node_prob[1] for node_prob in json_dict["node_prob"]}


