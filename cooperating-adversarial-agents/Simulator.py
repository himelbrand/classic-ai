import Env1
import Agent
from typing import List
from collections import defaultdict
import utils
global_file_name = "graph1.json"


class Simulator:
    
    deadline = 0

    def __init__(self,T=0):
        self.time_passed = 0
        self.agents_times = defaultdict(int)
        self.agents_pickup_times = defaultdict(int)
        self.T = T

    def initialize_env(self):
        """Environment initialization and agents creating function """

        # Environment source file : should be json file
        global global_file_name

        print("Reading the environment input file...")

        graph, people, Simulator.deadline = Env1.load_environment(global_file_name)

        print("Done reading the environment input file...\n\n")
        Simulator.deadline = utils.promptMenu(f'Input deadline is {Simulator.deadline}, do you want to ignore it?',{'Yes':float('inf'),'No':Simulator.deadline})
       
        print("\nNeed agents information...\n\n")

        # Agents creating
        Agent.Agent.restart_ids()
        game_type = utils.promptMenu('Choose type of game:',{'Zero-Sum':0,'Semi':1,'Fully':2})
    
        agents_list = []  # type: List [Agent]
        agent_locations = {}
        cutoff = utils.promptIntegerPositive('\nPlease enter cutoff')
        loc1 = utils.promptIntegerFromRange('\nPlease enter Agent 1 location (node number) ',graph.graph.keys())
        loc2 = utils.promptIntegerFromRange('\nPlease enter Agent 2 location (node number) ',graph.graph.keys())
        if game_type == 0:
            agents_list.append(Agent.MiniMaxAgent.create_agent(loc1,cutoff))
            agents_list.append(Agent.MiniMaxAgent.create_agent(loc2,cutoff))
        elif game_type == 1:
            agents_list.append(Agent.SemiCooperativeAgent.create_agent(loc1,cutoff))
            agents_list.append(Agent.SemiCooperativeAgent.create_agent(loc2,cutoff))
        elif game_type == 2:
            agents_list.append(Agent.FullyCooperativeAgent.create_agent(loc1,cutoff))
            agents_list.append(Agent.FullyCooperativeAgent.create_agent(loc2,cutoff))
        agent_locations = {agent.get_id(): [agent.location, agent.location, 0] for agent in agents_list}
        print("\n\nInitializing environment")

        env = Env1.Environment(graph=graph, agents_location=agent_locations, people_location=people, blocked_edges=[],deadline=self.deadline,agents=agents_list)
        env.initialize()
        return agents_list, env

    def simulation_loop(self, env: Env1.Environment, agent_list):
        """The main loop of the simulation : the simulator traverse through the agent list , and if it's not terminated ,
        the agent recieves the last observation from the environment and returns action . The actions is passed to the
        environment , it performs it and return the new state ( observation ) """
        is_finished = False

        # Get initial observation from the environment
        observation = env.get_observation({})

        # The main loop
        self.print_env(observation,[],init=True)
        agents_path = defaultdict(list)
        while not is_finished:
            actions = [None]
            
            for agent in agent_list:
                if len(agents_path[agent.get_id()]) == 0 or (agents_path[agent.get_id()][-1] != str(observation['agents_location'][agent.get_id()][0]) and agents_path[agent.get_id()][-1] != 'T'):
                    agents_path[agent.get_id()].append(str(observation['agents_location'][agent.get_id()][0]))
                self.time_passed += 1
                # Recieve action from the current agent
                action = agent.next_action(observation)
                self.agents_times[agent.get_id()] = self.time_passed + observation['expansions'][agent.get_id()]*self.T
                # Check whether current agent is not terminated
                if agent.is_agent_terminated(self.agents_times[agent.get_id()] > self.deadline):
                    actions.append(None)
                    self.print_env(observation,actions)
                    continue
                actions.append(action)
                
                # Apply action and receive the last observation
                observation = env.apply_action(action)
                if observation['collected']:
                    self.agents_pickup_times[agent.get_id()] = self.agents_times[agent.get_id()]
                if action['action_tag'] == 'terminate':
                    agent.is_agent_terminated(True)
                    agents_path[agent.get_id()].append('T')
                self.print_env(observation,actions)
            if self.is_simulation_finished(agent_list,observation):
                # is_finished = True
                break
        results = [(self.agents_pickup_times[agent.get_id()],env.people_collected[a.get_id()],a.type) for a in agent_list]
        print('Paths taken:')
        print(f'Agent1: {"->".join(agents_path[1])}')
        print(f'Agent2: {"->".join(agents_path[2])}')
        return results

    def is_simulation_finished(self,agent_list,observation):
        if self.time_passed > Simulator.deadline:
            print("Finishing the simulation : The deadline was reached")
            return True

        people_locations = [node for node, people in observation["people_location"].items() if people > 0]

        if people_locations == []:
            print("Finishing the simulation : All the people were collected")
            return True

        active_agents = [agent for agent in agent_list if not agent.is_agent_terminated()]

        if active_agents == []:
            print("Finishing the simulation : All the agents terminated")
            return True

        return False

    def print_env(self,observation,actions,init=False):
        if self.time_passed > self.deadline:
            return
        if init:
            print("================Initial Enviorment===============")
        else:
            print(f"\n\t\t================Time-step is {self.time_passed}/{self.deadline} it was Agent{1 if self.time_passed%2 else 2} move================\n")
        G = observation['graph']
        A = observation['agents_location']
        PC = observation['people_collected']
        for a in A:
            term = not init and a < len(actions) and actions[a] is None
            n1,n2,d = A[a]
            if term:
                if n1 == n2:
                    print(f'(Terminated) Agent {a} collected {PC[a]} people, current agent time is {self.time_passed} and is at Node {n1} ')
                else:
                    print(f'(Terminated) Agent {a} collected {PC[a]} people, current agent time is {self.time_passed} and was crossing edge ({n1},{n2}) and completed {G.get_weight(n1,n2)-d}/{G.get_weight(n1,n2)} of traversal')
                continue
            action = actions[a]['action_tag'].upper() if a < len(actions) else None
            blocked = actions[a]['action_details']['to_block'] if action == 'BLOCK' else ''
            
            if n1 == n2:
                print(f'(Active) Agent {a} collected {PC[a]} people, current agent time is {self.time_passed} and is at Node {n1}{f" - last action: {action} {blocked}" if action else ""}')
            else:
                print(f'(Active) Agent {a} collected {PC[a]} people, current agent time is {self.time_passed} and is crossing edge ({n1},{n2}) and completed {G.get_weight(n1,n2)-d}/{G.get_weight(n1,n2)} of traversal')
        print('\n\n\n')


if __name__ == '__main__':
    print('\n\nStarting HW2 Simulator!\n\n')
    i=1
    running = True
    while running:
        global_file_name = utils.promptMenu('What graph do you want to use? ',{'graph1':'graph1.json','graph2':'graph2.json'})
        print(f'Starting simulation #{i}')
        simulator = Simulator()
        agents, env = simulator.initialize_env()
        simulator.simulation_loop(env, agents)
        running = utils.promptMenu('\n\nDo you want to run another simulation? ',{'No':False,'Yes':True})
        i+=1
