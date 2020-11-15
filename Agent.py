from typing import Dict, List, Set, Tuple
import Graph as Gr

"""Parent type of all the agents"""

class Link:
    def __init__(self,prev,data):
        self.prev = prev
        self.data = data


class Agent:
    next_id = 0

    @classmethod
    def __get_unique_id(cls):
        # Get unique id to each agent
        Agent.next_id += 1

        return Agent.next_id

    def __init__(self, type: str):
        self.id = Agent.__get_unique_id()
        self.type = type
        self.is_terminated = False

    def get_id(self):
        return self.id

    def next_action(self, observation: Dict):
        """Main interface function of each agent - it receives the state  , and returns the action"""
        return {"action_tag": "no-op", "action_details": {}}

    def is_agent_terminated(self):
        return self.is_terminated


class HumanAgent(Agent):

    @classmethod
    def create_agent(cls):
        """Agent factory function"""
        return HumanAgent()

    def __init__(self):
        super().__init__("human")

    def next_action(self, observation: Dict):
        """Main interface function of each agent - it receives the state  , and returns the action"""
        print()
        print("**********************  Human agent  ***************************")

        # Check whether my previous action succeed , if not - terminate
        is_previous_succeed = observation["agents_last_action"][self.id]

        if not is_previous_succeed:
            self.is_terminated = True
            return {"action_tag": "no-op", "action_details": {"agent_id": self.id}}

        # Print the agent's current location ( if node1 == node2 and distance  = 0 , the agent is in node2 ( not on the edge ))
        current_location = observation["agents_location"][self.id]
        # Source  node
        node1 = current_location[0]
        # Destination node
        node2 = current_location[1]
        # Remaining distance
        distance = current_location[2]
        print("Agent {id} is now on the way from {node1} to {node2} , remaining distance {dist}"
              .format(id=self.get_id(), node1=node1, node2=node2, dist=distance))

        # Print the possible nodes to traverse from node2
        # TODO : check if the neighbors are blocked
        graph = observation["graph"]
        neigbours_and_weights = graph.get_neigbours_and_weights(node2)
        neigbours, _ = zip(*neigbours_and_weights)
        print("Node {node2} neighbours are : {neigh}".format(node2=node2, neigh=neigbours))

        # Ask for destination
        print("Please enter the distination node:")

        distanation = int(input())

        return {"action_tag": "traverse", "action_details": {"agent_id": self.id, "to": distanation}}


class SaboteurAgent(Agent):

    @classmethod
    def create_agent(cls):
        """Agent factory function"""
        print("Please enter V (the number of no-ops until block) for this saboteur agent:")
        V = int(input())
        return SaboteurAgent(V)

    def __init__(self, V):
        self.V = V
        self.remaining_time = V + 1
        super().__init__("saboteur")

    def next_action(self, observation: Dict):
        print()
        print("**********************  Sabateur agent  ***************************")

        """Main interface function of each agent - it receives the state  , and returns the action"""
        is_previous_succeed = observation["agents_last_action"][self.id]
        node1, node2, distance = observation["agents_location"][self.get_id()]

        # If the previous action failed - terminate and don't do anything ( no-op )
        if not is_previous_succeed:
            self.is_terminated = True
            print("My previous action failed , terminating .....")
            return {"action_tag": "no-op", "action_details": {"agent_id": self.id}}

        # If the agent is on the edge keep moving towards the destination (node2)
        if node1 != node2:
            print("I'm on the way from {node1} to {node2} (remaining distance {dist} ), keep going ...."
                  .format(node1=node1, node2=node2, dist=distance))
            return {"action_tag": "traverse", "action_details": {"agent_id": self.id, "to": node2}}

        # Reduce agent's internal counter
        self.remaining_time -= 1

        # Do nothing if the counter didn't arrive to 0
        if self.remaining_time > 0 or self.is_terminated:
            print("{time} rounds remained , doing nothing ".format(time=self.remaining_time))
            return {"action_tag": "no-op", "action_details": {"agent_id": self.id}}

        # If the counter is zero - find the neighbor with lowest edge wight and block it
        if self.remaining_time == 0:
            print(("Looking for the edge to block."))

            neibhor = self.find_closest_neighbor(observation)
            if neibhor is None:
                self.is_terminated = True
                print("Failed. Terminating .")
                return {"action_tag": "no-op", "action_details": {"agent_id": self.id}}

            print("Blocking the edge {edge}".format(edge=(node2, neibhor)))
            return {"action_tag": "block", "action_details": {"agent_id": self.id, "to_block": (node2, neibhor)}}

        # If the counter is -1 - find the neighbor with lowest edge wight and begin moving towards it
        if self.remaining_time == -1:
            print("Looking for the next node to go.")

            neibhor = self.find_closest_neighbor(observation)
            if neibhor is None:
                self.is_terminated = True
                print("Failed. Terminating .")
                return {"action_tag": "no-op", "action_details": {"agent_id": self.id}}
            self.remaining_time = self.V + 1

            print("Moving to node {node}".format(node=neibhor))
            return {"action_tag": "traverse", "action_details": {"agent_id": self.id, "to": neibhor}}

    def find_closest_neighbor(self, observation):
        """Find the neighbor with the lowest edge that is not blocked"""
        _, node2, _ = observation["agents_location"][self.get_id()]
        blocked_edges = observation["blocked_edges"]
        graph = observation["graph"]

        neighbors_and_weights = graph.get_neigbours_and_weights(node2)
        # Revert (neighbor, weight ) tuples and sort them
        weights_and_neighbors = sorted([(weigh, neib) for neib, weigh in neighbors_and_weights])

        # Traverse the neighbors (from the nearest one ) and check if it is not in blocked list , return the first that succeed
        for _, neighbor in weights_and_neighbors:
            if not (node2, neighbor) in blocked_edges:
                return neighbor
        return None


class GreedyAgent(Agent):

    @classmethod
    def create_agent(cls):
        """Agent factory function"""
        return GreedyAgent()

    def __init__(self):
        self.current_destination = None  # Current destination node : may be any node
        self.current_path = []  # List of the nodes that are remain to agent to pass
        self.traversing_in_progress = False  # Whether the agent is on the way somewhere
        super().__init__("greedy")

    def next_action(self, observation: Dict):

        print()
        print("**********************  Greedy agent  ***************************")
        is_previous_succeed = observation["agents_last_action"][self.id]

        # If the previous action failed - terminate and don't do anything ( no-op )
        if not is_previous_succeed or self.is_terminated:
            self.is_terminated = True
            return {"action_tag": "no-op", "action_details": {"agent_id": self.id}}

        # If the agent is on the way somewhere , find next action to send ( according to the path )
        # Or if the agent arrived to the destination return {}
        if self.traversing_in_progress:
            # Return the next traverse command
            next_action = self.next_traverse_action(observation)

            if next_action is None:
                self.is_terminated = True
                return {"action_tag": "no-op", "action_details": {"agent_id": self.id}}

            if next_action != {}:
                return next_action

        # Compute the next destination node ( the nearest node with people )
        self.compute_destination(observation)

        # If can't compute the next destination , terminate
        if self.current_destination is None:
            self.is_terminated = True
            return {"action_tag": "no-op", "action_details": {"agent_id": self.id}}

        # Return the next traverse command ( given currently computed destination )
        next_action = self.next_traverse_action(observation)

        return next_action

    def compute_destination(self, observation):
        """Find the closest node with people and update destination and path"""

        blocked_edges = observation["blocked_edges"]
        graph = observation["graph"]
        current_location = observation["agents_location"][self.get_id()]
        node2 = current_location[1]

        # Find nodes where there is more than one men
        people_locations = [node for node, people in observation["people_location"].items() if people > 0]

        temp_distance = float("inf")
        self.current_destination = None
        self.current_path = []

        for node in people_locations:
            distance, path = graph.get_shortest_path_Dijk(node2, node, blocked_edges)
            if distance < temp_distance:
                temp_distance = distance
                self.current_path = path
                self.current_destination = node

    def next_traverse_action(self, observation):
        node1, node2, distance = observation["agents_location"][self.get_id()]
        blocked_edges = observation["blocked_edges"]

        graph = observation["graph"]
        # If the agent is on the edge keep moving towards the destination ( node2)
        if node1 != node2:
            print("I'm on the way from {node1} to {node2} (remaining distance {dist} ), keep going ...."
                  .format(node1=node1, node2=node2, dist=distance))
            return {"action_tag": "traverse", "action_details": {"agent_id": self.id, "to": node2}}

        # If the agent arrived to the destination , return empty dict
        if node2 == self.current_destination or self.current_destination is None:
            print("Arrived to destination node {node2}".format(node2=self.current_destination))
            self.current_path = []
            self.traversing_in_progress = False
            return {}

        # If the agent arrived to some node but it's not a destination - keep moving to the next node in the path
        if node2 == self.current_path[0] and self.current_destination == self.current_path[-1]:
            print("I'm on the way to {node2} , remaining path is {path},currently passed {node1} keep going ...."
                  .format(node1=self.current_path[0], path=self.current_path, node2=self.current_destination))
            # TODO if the edge on the path is blocked recompute the path
            self.traversing_in_progress = True
            self.current_path = self.current_path[1:]

            return {"action_tag": "traverse", "action_details": {"agent_id": self.id, "to": self.current_path[0]}}

        # If the agent ( for some reason ) isn't on the path to the destination , recompute the path from the current location
        # And begin moving
        else:
            print("Something is wrong . Computing the path to destination node {node}".format(
                node=self.current_destination))
            self.current_path = graph.get_shortest_path_Dijk(node2, self.current_destination, blocked_edges)
            if self.current_path == []:
                return None
            print("Begin to move .")

            self.current_path = self.current_path[1:]
            print("I'm on the way to {node2} , remaining path is {path},currently passed {node1} keep going ...."
                  .format(node1=self.current_path[0], path=self.current_path, node2=self.current_destination))
            self.traversing_in_progress = True
            return {"action_tag": "traverse", "action_details": {"agent_id": self.id, "to": self.current_path[0]}}


class PlanningAgent(Agent):

    @classmethod
    def create_agent(cls):
        """Agent factory function"""
        return PlanningAgent()

    def __init__(self):
        self.current_destination = None  # Current destination node : may be any node
        self.current_path = []  # List of the nodes that are remain to agent to pass
        self.traversing_in_progress = False  # Whether the agent is on the way somewhere

        super().__init__("planning")

    def graph_reduction(self, observation):
        blocked_edges = observation["blocked_edges"]
        graph = observation["graph"]
        current_location = observation["agents_location"][self.get_id()]
        source_node = current_location[1]

        # Find nodes where there is more than one men
        people_locations = [node for node, people in observation["people_location"].items() if people > 0]
        people_locations_copy = people_locations.copy()

        new_edges = {}
        new_graph = {node: [] for node in people_locations}
        new_graph[source_node] = []
        while people_locations != []:
            for node in people_locations:
                weight, path = graph.get_shortest_path_Dijk(source_node, node, blocked=blocked_edges)
                if weight < float("inf") and len(set(path[1:-1]).intersection(people_locations_copy)) == 0:
                    new_graph[node].append(source_node)
                    new_graph[source_node].append(node)
                    new_edges[(min(node, source_node), max(node, source_node))] = weight


                #TODO: There is a problem that not all pathes will be presented ( for example two equal weight pathes )
                #TODO: The dijekstra should return all equally weighted pathes

            source_node = people_locations[0]
            people_locations = people_locations[1:]

        new_graph = Gr.Graph(new_graph, new_edges)
        sp_tree = new_graph.min_spanning_tree_kruskal([])
        return new_graph

    def next_action(self, observation: Dict):
        new_graph = self.graph_reduction(observation)
        """Main interface function of each agent - it receives the state  , and returns the action"""
        return {"action_tag": "no-op", "action_details": {}}


    def make_plan_A_star(self,problem,heuristic,limit,):
        open = [Link(None,self.initial_state(problem))]

        counter = 0

        g = 0

        while counter < limit:
            if open == []:
                return None
            node = open.pop(0)
            g = self.get_distance(node)
            new_nodes = self.expand(heuristic,node,problem)
            open.extend(new_nodes)
            open.sort(key= lambda x:x)


    def initial_state(self,problem):
        new_graph = self.graph_reduction(problem)

        return {"abstract_graph":new_graph,
                "my_location": problem["agents_location"][self.get_id()],
                "people_location": problem["people_location"],
                "current_distance":0,
                "f_value": None}



    def goal_test(self,node):
        data = node.data

        node_with_people = [n for n,p in data["people_location"] if p > 0]

        return node_with_people != []


    def expand(self,heuristic,parent_node,problem):

        data = parent_node.data

        parent_graph = data["abstract_graph"] #type: Gr.Graph
        _,current_location,_ = data["my_location"]

        neighb_and_weight = parent_graph.get_neigbours_and_weights(current_location)

        child_nodes_list = []
        new_people_location = data["people_location"]
        for child_node,weight in neighb_and_weight:
            ...



