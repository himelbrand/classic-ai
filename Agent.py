from typing import Dict , List, Set, Tuple
import Graph as Gr



class Agent:

    next_id = 0

    @classmethod
    def __get_unique_id(cls):
        Agent.next_id += 1

        return  Agent.next_id

    def __init__(self, type:str):
        self.id = Agent.__get_unique_id()
        self.type = type
        self.is_terminated = False

    def get_id(self):
        return self.id

    def next_action(self,observation : Dict):
        return {"action_tag":"no-op", "action_details":{}}

    def is_agent_terminated(self):
        return self.is_terminated

class HumanAgent(Agent):

    @classmethod
    def create_agent(cls):
        return HumanAgent()

    def __init__(self):
        super().__init__("human")


    def next_action(self,observation : Dict):
        print("**********************  Human agent  ***************************")
        is_previous_succeed = observation["agents_last_action"][self.id]


        #TODO Previous action failure


        current_location = observation["agents_location"][self.id]
        node1 = current_location[0]
        node2= current_location[1]
        distance = current_location[2]
        print("Agent {id} is now on the way from {node1} to {node2} , remaining distance {dist}"
              .format(id=self.get_id(),node1=node1,node2=node2,dist=distance))



        graph = observation["graph"]

        neigbours_and_weights = graph.get_neigbours_and_weights(node2)

        neigbours, _ = zip (*neigbours_and_weights)

        print("Node {node2} neighbours are : {neigh}".format(node2=node2,neigh=neigbours))



        print("Please enter the distination node:")

        distanation = int(input())

        return {"action_tag":"traverse","action_details":{"agent_id":self.id,"to":distanation}}


class SaboteurAgent(Agent):

    @classmethod
    def create_agent(cls):
        print("Please enter V (the number of no-ops until block) for this saboteur agent:")
        V = int(input())
        return SaboteurAgent(V)

    def __init__(self,V):
        self.V = V
        self.remaining_time = V + 1
        super().__init__("saboteur")


    def next_action(self,observation : Dict):


        is_previous_succeed = observation["agents_last_action"][self.id]
        node1, node2, distance = observation["agents_location"][self.get_id()]

        if not is_previous_succeed :
            self.is_terminated = True
            return {"action_tag": "no-op", "action_details": {"agent_id": self.id}}

        if node1 != node2:
            return {"action_tag":"traverse", "action_details":{"agent_id":self.id,"to":node2}}


        self.remaining_time -= 1
        if self.remaining_time > 0 or self.is_terminated:
            return {"action_tag":"no-op", "action_details":{"agent_id":self.id}}

        if self.remaining_time == 0 :
            neibhor = self.find_closest_neighbor(observation)
            if neibhor is None:
                self.is_terminated = True
                return {"action_tag": "no-op", "action_details": {"agent_id": self.id}}
            return {"action_tag":"block", "action_details":{"agent_id":self.id,"to_block":(node2,neibhor)}}

        if self.remaining_time ==-1 :
            neibhor = self.find_closest_neighbor(observation)
            if neibhor is None:
                self.is_terminated = True
                return {"action_tag": "no-op", "action_details": {"agent_id": self.id}}
            self.remaining_time = self.V + 1
            return {"action_tag":"traverse", "action_details":{"agent_id":self.id,"to":neibhor}}


    def find_closest_neighbor(self,observation):
        _ ,node2, _= observation["agents_location"][self.get_id()]
        blocked_edges = observation["blocked_edges"]

        graph = observation["graph"]

        neighbors_and_weights = graph.get_neigbours_and_weights(node2)

        weights_and_neighbors = sorted([(weigh,neib) for neib,weigh in neighbors_and_weights])

        for _,neighbor in weights_and_neighbors:
            if not (node2,neighbor) in blocked_edges:
                return neighbor
        return  None

class GreedyAgent(Agent):

    @classmethod
    def create_agent(cls):
        return GreedyAgent()

    def __init__(self):
        self.current_destination = None
        self.current_path = []
        self.traversing_in_progress = False
        super().__init__("greedy")

    def next_action(self,observation : Dict):
        print("**********************  Greedy agent  ***************************")
        is_previous_succeed = observation["agents_last_action"][self.id]


        if not is_previous_succeed or self.is_terminated:
            self.is_terminated = True
            return {"action_tag": "no-op", "action_details": {"agent_id": self.id}}

        if self.traversing_in_progress:
            next_action = self.next_traverse_action(observation)

            if next_action is None:
                self.is_terminated = True
                return {"action_tag":"no-op", "action_details":{"agent_id":self.id}}

            if next_action != {}:
                return next_action

        self.compute_destination(observation)

        if self.current_destination is None:
            self.is_terminated = True
            return {"action_tag": "no-op", "action_details": {"agent_id": self.id}}

        next_action = self.next_traverse_action(observation)

        return next_action

    def compute_destination(self,observation):

        blocked_edges = observation["blocked_edges"]
        people_locations = [node for node,people in observation["people_location"].items() if people > 0]
        graph = observation["graph"]
        current_location = observation["agents_location"][self.get_id()]
        node2 = current_location[1]

        temp_distance = float("inf")
        self.current_destination = None
        self.current_path = []
        for node in people_locations:
            distance,path = graph.get_shortest_path_Dijk(node2,node,blocked_edges)
            if distance < temp_distance:
                temp_distance = distance
                self.current_path = path
                self.current_destination = node


    def next_traverse_action(self,observation):
        node1,node2,distance = observation["agents_location"][self.get_id()]
        blocked_edges = observation["blocked_edges"]

        graph = observation["graph"]
        # The Agent is on the edge : keep going towards the node2
        if node1 != node2:
            return {"action_tag":"traverse","action_details":{"agent_id":self.id,"to":node2}}

        if node2 == self.current_destination or self.current_destination is None:
            self.current_path = []
            self.traversing_in_progress = False
            return {}

        if node2 == self.current_path[0] and self.current_destination == self.current_path[-1]:
            self.traversing_in_progress = True
            self.current_path = self.current_path[1:]

            return {"action_tag":"traverse","action_details":{"agent_id":self.id,"to":self.current_path[0]}}
        else:
            self.current_path = graph.get_shortest_path_Dijk(node2,self.current_destination,blocked_edges)
            if self.current_path == []:
                return None
            self.current_path = self.current_path[1:]
            self.traversing_in_progress = True
            return {"action_tag":"traverse","action_details":{"agent_id":self.id,"to":self.current_path[0]}}





