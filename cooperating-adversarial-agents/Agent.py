from typing import Dict, List, Set, Tuple
import Graph as Gr
import distance_algs as algo
from GameTree import GameTree
"""Parent type of all the agents"""


class Link:
    def __init__(self, prev, data):
        self.prev = prev
        self.data = data


class Agent:
    next_id = 0

    @classmethod
    def __get_unique_id(cls):
        # Get unique id to each agent
        Agent.next_id += 1

        return Agent.next_id
    
    @classmethod
    def restart_ids(cls):
        # Get unique id to each agent
        Agent.next_id = 0

    def __init__(self, agent_type: str,loc):
        self.id = Agent.__get_unique_id()
        self.type = agent_type
        self.is_terminated = False
        self.score = 0
        self.t = 0
        self.location = loc
        self.traversing = False


    def get_id(self):
        return self.id

    def next_action(self, observation: Dict):
        """Main interface function of each agent - it receives the state  , and returns the action"""
        return {"action_tag": "no-op", "action_details": {'agent_id':self.id}}

    def is_agent_terminated(self,terminate=False):
        if terminate:
            self.is_terminated = True
        return self.is_terminated
            
class MiniMaxAgent(Agent):
    @classmethod
    def create_agent(cls,loc,cutoff):
        """Agent factory function"""
        return MiniMaxAgent(loc,cutoff)


    def __init__(self,loc,cutoff):
        super().__init__("minimax",loc)
        self.cutoff = cutoff
        self.visited = set()
    def next_action(self, observation: Dict):
        # Check whether my previous action succeed , if not - terminate
        is_previous_succeed = observation["agents_last_action"][self.id]
        second_agent = observation['agents'][self.id%2]
        observed_state = GameTree.generateGameState(self,second_agent,observation)
        if str(observed_state) in self.visited:
            print('WTF')
        self.visited.add(str(observed_state))
        if not is_previous_succeed:
            return {"action_tag": "terminate", "action_details": {"agent_id": self.id}}
        current_location = observation["agents_location"][self.id]
        
        # Destination node and remaining distance
        _,dest,rd = current_location
       
        if rd:
            self.traversing = True
            return {"action_tag": "traverse", "action_details": {"agent_id": self.id, "to": dest}}
        
        if not is_previous_succeed or self.is_terminated:
            self.is_terminated = True
            return {"action_tag": "terminate", "action_details": {"agent_id": self.id}}
        if second_agent.id ==self.id:
            print('WTF!!!',self.id)
        tree = GameTree(self,second_agent,observation,self.cutoff)
        self.visited.add(str(tree.state))
        a,dest = tree.alpha_beta_decision(self.visited)
        return {'action_tag':a,'action_details':{'agent_id':self.id,'to':dest}}

class SemiCooperativeAgent(Agent):
    @classmethod
    def create_agent(cls,loc,cutoff):
        """Agent factory function"""
        return SemiCooperativeAgent(loc,cutoff)


    def __init__(self,loc,cutoff):
        super().__init__("Semi-cooperative",loc)
        self.cutoff = cutoff
        self.visited = set()
    
    def next_action(self, observation: Dict):
        # Check whether my previous action succeed , if not - terminate
        is_previous_succeed = observation["agents_last_action"][self.id]
        second_agent = observation['agents'][self.id%2]
        observed_state = GameTree.generateGameState(self,second_agent,observation)
        if str(observed_state) in self.visited:
            print('WTF')
        self.visited.add(str(observed_state))
        if not is_previous_succeed:
            return {"action_tag": "terminate", "action_details": {"agent_id": self.id}}
        current_location = observation["agents_location"][self.id]
        
        # Destination node and remaining distance
        _,dest,rd = current_location
       
        if rd:
            self.traversing = True
            return {"action_tag": "traverse", "action_details": {"agent_id": self.id, "to": dest}}
        
        if not is_previous_succeed or self.is_terminated:
            self.is_terminated = True
            return {"action_tag": "terminate", "action_details": {"agent_id": self.id}}
        
        tree = GameTree(self,second_agent,observation,self.cutoff)
        self.visited.add(str(tree.state))
        a,dest = tree.semi_decision(self.visited)
        
        return {'action_tag':a,'action_details':{'agent_id':self.id,'to':dest}}

class FullyCooperativeAgent(Agent):
    @classmethod
    def create_agent(cls,loc,cutoff):
        """Agent factory function"""
        return FullyCooperativeAgent(loc,cutoff)


    def __init__(self,loc,cutoff):
        super().__init__("Fully-cooperative",loc)
        self.cutoff = cutoff
        self.visited = set()
    
    def next_action(self, observation: Dict):
        # Check whether my previous action succeed , if not - terminate
        is_previous_succeed = observation["agents_last_action"][self.id]
        second_agent = observation['agents'][self.id%2]
        observed_state = GameTree.generateGameState(self,second_agent,observation)
        if str(observed_state) in self.visited:
            print('WTF')
        self.visited.add(str(observed_state))
        if not is_previous_succeed:
            return {"action_tag": "terminate", "action_details": {"agent_id": self.id}}
        current_location = observation["agents_location"][self.id]
        
        # Destination node and remaining distance
        _,dest,rd = current_location
       
        if rd:
            self.traversing = True
            return {"action_tag": "traverse", "action_details": {"agent_id": self.id, "to": dest}}
        
        if not is_previous_succeed or self.is_terminated:
            self.is_terminated = True
            return {"action_tag": "terminate", "action_details": {"agent_id": self.id}}
        
        tree = GameTree(self,second_agent,observation,self.cutoff)
        self.visited.add(str(tree.state))
        a,dest = tree.fully_decision(self.visited)
        return {'action_tag':a,'action_details':{'agent_id':self.id,'to':dest}}

class HumanAgent(Agent):

    @classmethod
    def create_agent(cls,loc):
        """Agent factory function"""
        return HumanAgent(loc)

    def __init__(self,loc):
        super().__init__("human",loc)

    def next_action(self, observation: Dict):
        """Main interface function of each agent - it receives the state  , and returns the action"""
        print()

        # Check whether my previous action succeed , if not - terminate
        is_previous_succeed = observation["agents_last_action"][self.id]
        if not is_previous_succeed:
            return {"action_tag": "terminate", "action_details": {"agent_id": self.id}}

        current_location = observation["agents_location"][self.id]
        
        # Destination node and remaining distance
        _,dest,rd = current_location
       
        if rd:
            return {"action_tag": "traverse", "action_details": {"agent_id": self.id, "to": dest}}

        # Print the possible nodes to traverse from dest
        # TODO : check if the neighbors are blocked
        graph = observation["graph"]
        neigbours_and_weights = graph.get_neigbours_and_weights(dest)
        neigbours, _ = zip(*neigbours_and_weights)


        print("**********************  Human agent  ***************************")
        print(f"Agent {self.id} is at Node {dest}, possible neighbours are : {neigbours}")
        # Ask for destination
        print("Please enter the distination node or 0 for termination:")

        distanation = int(input())

        return {"action_tag": "traverse",
                "action_details": {"agent_id": self.id, "to": distanation}} if distanation else {
            "action_tag": "terminate", "action_details": {"agent_id": self.id}}


class SaboteurAgent(Agent):

    @classmethod
    def create_agent(cls,loc):
        """Agent factory function"""
        while True:
            try:
                V = int(input("Please enter V (the number of no-ops until block) for this saboteur agent: "))
                break
            except:
                print('Invalid choice, must be an integer...')
        return SaboteurAgent(V,loc)

    def __init__(self, V,loc):
        self.V = V
        self.remaining_time = V + 1
        super().__init__("saboteur",loc)

    def next_action(self, observation: Dict):
        """Main interface function of each agent - it receives the state  , and returns the action"""

        is_previous_succeed = observation["agents_last_action"][self.id]
        origin, dest, distance = observation["agents_location"][self.id]

        # If the previous action failed - terminate and don't do anything ( no-op )
        if not is_previous_succeed:
            # self.is_terminated = True
            print("My previous action failed , terminating .....")
            return {"action_tag": "terminate", "action_details": {"agent_id": self.id}}

        # If the agent is on the edge keep moving towards the destination (dest)
        if origin != dest:
            print("I'm on the way from {origin} to {dest} (remaining distance {dist} ), keep going ...."
                  .format(origin=origin, dest=dest, dist=distance))
            return {"action_tag": "traverse", "action_details": {"agent_id": self.id, "to": dest}}

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
                # self.is_terminated = True
                # print("Failed. Terminating .")
                return {"action_tag": "terminate", "action_details": {"agent_id": self.id}}

            print("Blocking the edge {edge}".format(edge=(dest, neibhor)))
            return {"action_tag": "block", "action_details": {"agent_id": self.id, "to_block": (dest, neibhor)}}

        # If the counter is -1 - find the neighbor with lowest edge wight and begin moving towards it
        if self.remaining_time == -1:
            print("Looking for the next node to go.")

            neibhor = self.find_closest_neighbor(observation)
            if neibhor is None:
                # self.is_terminated = True
                # print("Failed. Terminating .")
                return {"action_tag": "terminate", "action_details": {"agent_id": self.id}}
            self.remaining_time = self.V + 1

            print("Moving to node {node}".format(node=neibhor))
            return {"action_tag": "traverse", "action_details": {"agent_id": self.id, "to": neibhor}}

    def find_closest_neighbor(self, observation):
        """Find the neighbor with the lowest edge that is not blocked"""
        _, dest, _ = observation["agents_location"][self.id]
        blocked_edges = observation["blocked_edges"]
        graph = observation["graph"]

        neighbors_and_weights = graph.get_neigbours_and_weights(dest)
        # Revert (neighbor, weight ) tuples and sort them
        weights_and_neighbors = sorted([(weigh, neib) for neib, weigh in neighbors_and_weights])

        # Traverse the neighbors (from the nearest one ) and check if it is not in blocked list , return the first that succeed
        for _, neighbor in weights_and_neighbors:
            if not (dest, neighbor) in blocked_edges:
                return neighbor
        return None


class GreedyAgent(Agent):

    @classmethod
    def create_agent(cls,loc):
        """Agent factory function"""
        return GreedyAgent(loc)

    def __init__(self,loc):
        self.current_destination = None  # Current destination node : may be any node
        self.current_path = []  # List of the nodes that are remain to agent to pass
        self.traversing_in_progress = False  # Whether the agent is on the way somewhere
        super().__init__("greedy",loc)

    def next_action(self, observation: Dict):

        print()
        # print("**********************  Greedy agent  ***************************")
        is_previous_succeed = observation["agents_last_action"][self.id]

        # If the previous action failed - terminate and don't do anything ( no-op )
        if not is_previous_succeed or self.is_terminated:
            # self.is_terminated = True
            return {"action_tag": "terminate", "action_details": {"agent_id": self.id}}

        # If the agent is on the way somewhere , find next action to send ( according to the path )
        # Or if the agent arrived to the destination return {}
        if self.traversing_in_progress:
            # Return the next traverse command
            next_action = self.next_traverse_action(observation)

            if next_action is None:
                # self.is_terminated = True
                return {"action_tag": "terminate", "action_details": {"agent_id": self.id}}

            if next_action != {}:
                return next_action

        # Compute the next destination node ( the nearest node with people )
        self.compute_destination(observation)

        # If can't compute the next destination , terminate
        if self.current_destination is None:
            # self.is_terminated = True
            return {"action_tag": "terminate", "action_details": {"agent_id": self.id}}

        # Return the next traverse command ( given currently computed destination )
        next_action = self.next_traverse_action(observation)

        return next_action

    def compute_destination(self, observation):
        """Find the closest node with people and update destination and path"""

        blocked_edges = observation["blocked_edges"]
        graph = observation["graph"]
        current_location = observation["agents_location"][self.id]
        dest = current_location[1]

        # Find nodes where there is more than one men
        people_locations = [node for node, people in observation["people_location"].items() if
                            people > 0 and dest != node]

        temp_distance = float("inf")
        self.current_destination = None
        self.current_path = []

        for node in people_locations:
            distance, path = algo.get_shortest_path_Dijk(graph,dest, node, blocked_edges)
            if distance < temp_distance:
                temp_distance = distance
                self.current_path = path
                self.current_destination = node

    def next_traverse_action(self, observation):
        origin, dest, _ = observation["agents_location"][self.id]
        blocked_edges = observation["blocked_edges"]

        graph = observation["graph"]
        # If the agent is on the edge keep moving towards the destination ( dest)
        if origin != dest:
            return {"action_tag": "traverse", "action_details": {"agent_id": self.id, "to": dest}}

        # If the agent arrived to the destination , return empty dict
        if dest == self.current_destination or self.current_destination is None:
            self.current_path = []
            self.traversing_in_progress = False
            return {}

        # If the agent arrived to some node but it's not a destination - keep moving to the next node in the path
        if dest == self.current_path[0] and self.current_destination == self.current_path[-1]:
            # TODO if the edge on the path is blocked recompute the path
            self.traversing_in_progress = True
            self.current_path = self.current_path[1:]

            return {"action_tag": "traverse", "action_details": {"agent_id": self.id, "to": self.current_path[0]}}

        # If the agent ( for some reason ) isn't on the path to the destination , recompute the path from the current location
        # And begin moving
        else:
            self.current_path = algo.get_shortest_path_Dijk(graph,dest, self.current_destination, blocked_edges)
            if self.current_path == []:
                return None

            self.current_path = self.current_path[1:]
            self.traversing_in_progress = True
            return {"action_tag": "traverse", "action_details": {"agent_id": self.id, "to": self.current_path[0]}}


class SearchAgent(Agent):

    @classmethod
    def create_agent(cls,loc):
        """Agent factory function"""
        options = {'G':('greedy',1),'A':('A*',10000),'S':('simple_RT_A*',None)}
        while True:
            answer = input('Possible search agent sub-class:\n(G) greedy\n(A) A*\n(S) simple RT A*\n\nPlease choose an option: ').upper()
            if answer not in options:
                print('Invalid choice, please choose one of: G, A, S')
                continue
            name,limit = options[answer]
            if limit is None:
                while True:
                    try:
                        limit = int(input('Please enter a limit parameter for this agent: '))
                        break
                    except:
                        print('Invalid input, limit must be an integer...')
            break
        return SearchAgent(name,loc,limit=limit)
            
            

    def __init__(self, agent_type,loc, limit=10000):
        self.current_destination = None  # Current destination node : may be any node
        self.current_path = []  # List of the nodes that are remain to agent to pass
        self.traversing_in_progress = False  # Whether the agent is on the way somewhere
        self.destination_bank = []  # The sequence of nodes to visit that were returned by planner
        self.limit = limit
        self.expansions = 0
        self.last_expansions = 0

        super().__init__("search_{}".format(agent_type),loc)

    def next_action(self, observation: Dict):
        self.last_expansions = 0
        is_previous_succeed = observation["agents_last_action"][self.id]

        # If the previous action failed - terminate and don't do anything ( no-op )
        if not is_previous_succeed or self.is_terminated:
            # self.is_terminated = True
            return {"action_tag": "terminate", "action_details": {"agent_id": self.id,'expansions':self.last_expansions}}

        # If the agent is on the way somewhere , find next action to send ( according to the path )
        # Or if the agent arrived to the destination return {}
        if self.traversing_in_progress:
            # Return the next traverse command
            next_action = self.next_traverse_action(observation)

            if next_action is None:
                # self.is_terminated = True
                return {"action_tag": "terminate", "action_details": {"agent_id": self.id,'expansions':self.last_expansions}}

            if next_action != {}:
                return next_action

        # Compute the next destination node ( the nearest node with people )
        # Or a sequence of destinations ( verteces to visit )
        self.compute_destination(observation)

        # If can't compute the next destination , terminate
        if self.current_destination is None:
            # self.is_terminated = True
            return {"action_tag": "terminate", "action_details": {"agent_id": self.id,'expansions':self.last_expansions}}

        # Return the next traverse command ( given currently computed destination )
        next_action = self.next_traverse_action(observation)

        return next_action

    def compute_destination(self, observation):
        """Find the closest node with people and update destination and path"""

        blocked_edges = observation["blocked_edges"]
        graph = observation["graph"]
        current_location = observation["agents_location"][self.id]
        dest = current_location[1]

        self.current_destination = None
        self.current_path = []

        if len(self.destination_bank) == 0:
            self.destination_bank = self.make_plan_A_star(observation, SearchAgent.MST_heuristic, self.limit)
            if self.destination_bank is None:
                self.current_destination = None
                return
            self.destination_bank.pop(0)
            if 'RT' in self.type:
                self.destination_bank = self.destination_bank[:1]

        

        self.current_destination = self.destination_bank.pop(0)

        _, self.current_path = algo.get_shortest_path_Dijk(graph,dest, self.current_destination, blocked_edges)

    def next_traverse_action(self, observation):
        origin, dest, _ = observation["agents_location"][self.id]
        blocked_edges = observation["blocked_edges"]

        graph = observation["graph"]
        # If the agent is on the edge keep moving towards the destination ( dest)
        if origin != dest:
            return {"action_tag": "traverse", "action_details": {"agent_id": self.id, "to": dest, 'expansions':self.last_expansions}}
        # If the agent arrived to the destination , return empty dict
        if dest == self.current_destination or self.current_destination is None:
            self.current_path = []
            self.traversing_in_progress = False
            return {}

        # If the agent arrived to some node but it's not a destination - keep moving to the next node in the path
        if dest == self.current_path[0] and self.current_destination == self.current_path[-1]:
            # TODO if the edge on the path is blocked recompute the path
            self.traversing_in_progress = True
            self.current_path = self.current_path[1:]

            return {"action_tag": "traverse", "action_details": {"agent_id": self.id, "to": self.current_path[0],'expansions':self.last_expansions}}

        # If the agent ( for some reason ) isn't on the path to the destination , recompute the path from the current location
        # And begin moving
        else:
            self.current_path = algo.get_shortest_path_Dijk(graph,dest, self.current_destination, blocked_edges)
            if self.current_path == []:
                return None

            self.current_path = self.current_path[1:]
            self.traversing_in_progress = True
            return {"action_tag": "traverse", "action_details": {"agent_id": self.id, "to": self.current_path[0],'expansions':self.last_expansions}}

    def graph_reduction(self, observation):
        blocked_edges = observation["blocked_edges"]
        graph = observation["graph"]
        current_location = observation["agents_location"][self.id]
        source_node = current_location[1]

        # Find nodes where there is more than one men
        people_locations = [node for node, people in observation["people_location"].items() if people > 0 and node != source_node
                                                                                                ]

        people_locations_copy = people_locations.copy()

        new_edges = {}
        new_graph = {node: [] for node in people_locations}
        new_graph[source_node] = []
        while people_locations != []:
            for node in people_locations:
                weight, path = algo.get_shortest_path_Dijk(graph,source_node, node, blocked=blocked_edges)
                if weight < float("inf") and len(set(path[1:-1]).intersection(people_locations_copy)) == 0:
                    new_graph[node].append(source_node)
                    new_graph[source_node].append(node)
                    new_edges[(min(node, source_node), max(node, source_node))] = weight

                # TODO: There is a problem that not all pathes will be presented ( for example two equal weight pathes )
                # TODO: The dijekstra should return all equally weighted pathes

            source_node = people_locations[0]
            people_locations = people_locations[1:]

        new_graph = Gr.Graph(new_graph, new_edges)

        return new_graph

    def people_remaining(self,problem):
        agents_loc = problem['agents_location']
        people_loc = problem['people_location']
        origin,dest,rd = agents_loc[self.id]
        to_ignore = origin if origin == dest and rd == 0 else -1
        people_count = sum([people_loc[n] for n in people_loc if n != to_ignore])
        return people_count


    def make_plan_A_star(self, problem, heuristic, limit):
        fringe = [Link(None, self.initial_state(problem))]
        if self.people_remaining(problem) == 0:
            self.last_expansions = 0
            return None
        counter = 0
        node = fringe.pop(0)
        while counter < limit and not self.goal_test(node):
            new_nodes = self.expand(heuristic, node)
            fringe.extend(new_nodes)
            fringe.sort(key=lambda link: link.data["f_value"])
            counter += 1
            if len(fringe) == 0:
                self.expansions += counter
                self.last_expansions = counter
                return None
            node = fringe.pop(0)
           

        path = []
        self.expansions += counter
        self.last_expansions = counter
        while node != None:
            _, vertex, _ = node.data["agents_location"][self.id]
            path.append(vertex)
            node = node.prev
        path.reverse()
        return path

    def initial_state(self, problem):
        new_graph = self.graph_reduction(problem)

        return {"graph": new_graph,
                "agents_location": problem["agents_location"],
                "people_location": problem["people_location"],

                "g_value": 0,
                "f_value": None}

    def goal_test(self, node):
        data = node.data

        node_with_people = [n for n, p in data["people_location"].items() if p > 0]

        return node_with_people == []

    def expand(self, heuristic, parent_node):
        data = parent_node.data

        parent_graph = data["graph"]  # type: Gr.Graph
        _, current_location, _ = data["agents_location"][self.id]
        parent_g_value = data["g_value"]
        neighb_and_weight = parent_graph.get_neigbours_and_weights(current_location)

        child_nodes_list = []

        for neib, weight in neighb_and_weight:
            new_people_location = data["people_location"].copy()
            new_people_location[neib] = 0

            new_agent_location = {}
            new_agent_location[self.id] = [neib, neib, 0]

            new_graph = self.graph_reduction({"graph": parent_graph,
                                              "people_location": new_people_location,
                                              "agents_location": new_agent_location,
                                              "blocked_edges": []})
            new_state = {"graph": new_graph,
                         "agents_location": new_agent_location,
                         "people_location": new_people_location,

                         "g_value": parent_g_value + weight,
                         "f_value": None}
            h_value = heuristic(self, new_state)

            if 'greedy' in self.type:
                new_state["f_value"] = h_value
            else:
                new_state["f_value"] = new_state["g_value"] + h_value

            child_nodes_list.append(Link(parent_node, new_state))

        return child_nodes_list

    def MST_heuristic(self, state):
        graph = state["graph"]  # type: Gr.Graph
        _, weight = algo.min_spanning_tree_kruskal(graph,[])

        return weight

    def MST_heuristic_ppl(self, state):
        graph = state["graph"]  # type: Gr.Graph
        G = Gr.Graph(graph.graph.copy(), graph.weights.copy())
        ppl_sum = sum([state["people_location"][k] for k in state["people_location"]])
        edges_sum = 0
        for e in G.weights:
            n1, n2 = e
            p1, p2 = state["people_location"][n1], state["people_location"][n2]
            G.weights[e] += ppl_sum - p1 - p2
            edges_sum += G.weights[e]
        for e in G.weights:
            G.weights[e] /= edges_sum

        _, weight = algo.min_spanning_tree_kruskal(G,[])

        return weight
