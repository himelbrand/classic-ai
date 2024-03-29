from typing import Dict, List, Set, Tuple
import Graph as Gr
import json


def load_environment(file_name):
    with open(file_name) as f:
        json_dict = json.load(f)

    deadline = json_dict["deadline"]

    nodes = list(range(1, json_dict["nodes_num"] + 1))

    people_location = {node_people[0]: node_people[1] for node_people in json_dict["node_people"]}

    # TODO Check consistancy

    edges_weights = json_dict["n1_n2_weight"]
    graph = {node: [] for node in nodes}
    weights = {}

    for node1, node2, weight in edges_weights:
        graph[node1].append(node2)
        graph[node2].append(node1)
        weights[(min(node1, node2), max(node1, node2))] = weight

    graph = Gr.Graph(graph, weights)

    return graph, people_location, deadline


class Environment:

    def __init__(self, graph, agents_location, people_location, blocked_edges,deadline,agents):
        self.graph = graph  # type: Gr.Graph                      # Initial graph of the world
        self.deadline = deadline
        self.agents_location = agents_location  # type: Dict[int,List[int,int,int]]   # Current location of the agents :
        # key - agent ID , value is triple ( node1,node2,distance)
        # if node1 == node2 and distance = 0 , the agent is in node2 ,
        # else the agent is on the edge (node1,node2) at distance X from node2
        self.agents_expansions = {a:0 for a in agents_location}
        self.people_location = people_location  # type: Dict[int,int]                 # People location : key - node , value - number of people
        self.blocked_edges = blocked_edges  # type: List[(int,int)]               # Edges that were blocked
        self.agents = agents
        self.people_collected = {}  # type: Dict[int,int]                 # People collected by each agent , key - agent ID , value - number of people
        self.agents_last_action = {}  # type: Dict[int,bool]                # The result of each agent's last action , True if the actions succeed , False otherwise

        # Dictionary with functions that should be activated for each action
        self.actions_reactions = {"traverse": self.traverse, "no-op": self.no_op_action, "block": self.block, "terminate":self.terminate_action}

    def initialize(self):
        for agent in self.agents_location.keys():
            self.people_collected[agent] = 0
            self.agents_last_action[agent] = True

    def apply_action(self, action):
        """The main interface function
        It receives an action ( dictionary ) of the following format {\"action_tag\" : ..., \"action_details\" : ....} and
        invokes the function from action_reaction dict according to action_tag
        The function returns current environment state"""
        agent = action['action_details']['agent_id']
        current_node = self.agents_location[agent][0]
        people_collected = self.people_location.get(current_node, 0)
        self.people_location[current_node] = 0
        self.people_collected[agent] += people_collected
        self.agents[agent-1].score += people_collected
        self.agents[agent-1].location = current_node
        self.agents[agent-1].t += 1
        self.agents[agent%2].t += 1
        # self.agents_location[agent%2+1][2] = max(self.agents_location[agent%2+1][2]-1,0)
        # if self.agents_location[agent%2+1][2] == 0:
        #     self.agents_location[agent%2+1][0] = self.agents_location[agent%2+1][1]
        #     self.agents[agent%2].traversing = False
        #     self.agents[agent%2].location = self.agents_location[agent%2+1][1]

        if 'expansions' in action["action_details"]:
            self.agents_expansions[action["action_details"]['agent_id']] += action["action_details"]['expansions']
        resulting_observ = self.actions_reactions[action["action_tag"]](action["action_details"])
        resulting_observ['collected'] = people_collected
        return resulting_observ

    def traverse(self, action_details: Dict):
        """Function that is invoked when traverse action is performed by the agent """
        agent = action_details["agent_id"]
        self.agents[agent-1].traversing = True
        # distanation node
        dest_node = action_details["to"]

        # TODO add checks for from and to nodes

        node1, node2, distance = self.agents_location[agent]
        # people_collected = 0
        
        # If the agent is in node ( not on the edge ) check if the distination node is its neighbor
        if node1 == node2 and self.graph.is_neighbours(node1, dest_node) and not (node2,dest_node) in self.blocked_edges :
            # Get (node1,dest_node) edge weight

            edge_weight = self.graph.get_weight(node1, dest_node)

            # Move the agent into the edge (node1,dest_node)
            distance = edge_weight - 1
            self.agents_location[agent] = [node1, dest_node, distance]
            action_succeed = True

        # If the agent is already inside the edge , check whether destination node is correct
        elif node1 != node2 and node2 == dest_node:

            # Move the agent one step on the edge
            distance -= 1
            self.agents_location[agent][2] = distance

            action_succeed = True
        else:
            # If the destination node is wrong
            action_succeed = False
            # TODO write warning

        # If the agent arrived to some node , collect all the people there and change the location from [node1,node2,X]
        # to [dest_node,dest_node,0]
        if distance == 0 and action_succeed:
            self.agents_location[agent] = [dest_node, dest_node, 0]
            self.agents[agent-1].traversing = False
            self.agents[agent-1].location = dest_node
            action_succeed = True

        self.agents_last_action[agent] = action_succeed

        new_observation = self.get_observation({})

        return new_observation

    def block(self, action_details):
        """The function for block action """
        agent = action_details["agent_id"]
        node1, node2, _ = self.agents_location[agent]

        node_block1, node_block2 = action_details["to_block"]

        # Check whether the agent is in node ( not on the edge ) and that the edge to block is connected to this node
        if node1 == node2 and node1 in [node_block1, node_block2]:
            self.blocked_edges.extend([(node_block1, node_block2), (node_block2, node_block1)])
            action_succeed = True
        else:
            action_succeed = False

        self.agents_last_action[agent] = action_succeed
        new_observation = self.get_observation({})

        return new_observation

    def no_op_action(self, action_details: Dict):
        """No-op action"""

        agent = action_details["agent_id"]
        self.agents_last_action[agent] = True
        new_observation = self.get_observation({})
        return new_observation
    
    def terminate_action(self, action_details: Dict):
        """Terminate action"""

        agent = action_details["agent_id"]
        self.agents[agent-1].is_terminated = True
        self.agents_last_action[agent] = False
        new_observation = self.get_observation({})
        return new_observation

    def get_observation(self, args: Dict):
        """Return the world's current state"""
        new_observation = {"graph": self.graph,
                           "agents_location": self.agents_location,
                           "people_location": self.people_location,
                           "blocked_edges": self.blocked_edges,
                           "people_collected": self.people_collected,
                           "agents_last_action": self.agents_last_action,
                           'expansions': self.agents_expansions,
                           'agents':self.agents,
                           'deadline':self.deadline}

        new_observation.update(args)
        return new_observation

