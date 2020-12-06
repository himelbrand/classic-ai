import Env1
import Agent
from typing import List
from collections import defaultdict
global_file_name = "graph4.json"


class Simulator:
    agent_init_functions = {"human": Agent.HumanAgent.create_agent,
                            "greedy": Agent.GreedyAgent.create_agent,
                            "sabateur": Agent.SaboteurAgent.create_agent,
                            "search":Agent.SearchAgent.create_agent}  # Agent type and the function that creates particular agent
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
        while True:
            ans = input(f'Input deadline is {Simulator.deadline}, do you want to ignore it (Y/N) ? ').upper()
            if ans == 'Y':
                Simulator.deadline = float('inf')
                break
            elif ans == 'N':
                break
            else:
                print('Invalid input, choose one of: Y, N')

        print("\nNeed agents information...\n\n")

        # Agents creating
        Agent.Agent.restart_ids()
        while True:
            try:
                ans = input("How many agents do you want to run? ")
                agent_num = int(ans)
                break
            except:
                print('You must enter an integer...\ntry again...')

        agents_list = []  # type: List [Agent]
        agent_locations = {}

        for i in range(1, agent_num + 1):
            agent_types = list(Simulator.agent_init_functions.keys())
            options = [f'({i}) - {t}' for i,t in enumerate(agent_types)]
            short = {str(i): t for i,t in enumerate(agent_types)}
            while True:
                print("Agent {} : Please enter the agent type:\n{}".format(i, '\n'.join(options)))
                agent_type = input('Your choice: ')
                if len(agent_type) == 1 and agent_type in short:
                    agent_type = short[agent_type]
                    break
                if len(agent_type) > 1 and agent_type in agent_types:
                    break
                print(f'Your choice of: "{agent_type}" is invalid, pick again from list (either number or complete name)')

            while True:
                try:
                    ans = input("Please enter agent location (node number): ")
                    agent_location = int(ans)
                    if agent_location not in graph.graph:
                        print(f'Your pick of {agent_location} is invalid must be on of { ", ".join([str(n) for n in graph.graph]) }')
                    else:
                        break
                except:
                    print('You must enter a valid node number of type int...')

            # Invoking agent initialization function ( according to agent type )
            agent = Simulator.agent_init_functions[agent_type]()
            agents_list.append(agent)
            agent_locations[agent.get_id()] = [agent_location, agent_location, 0]

        print("\n\nInitializing environment")

        env = Env1.Environment(graph=graph, agents_location=agent_locations, people_location=people, blocked_edges=[])
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
        while not is_finished:
            actions = [None]
            self.time_passed += 1
            for agent in agent_list:
                
                # Recieve action from the current agent
                action = agent.next_action(observation)
                self.agents_times[agent.get_id()] = self.time_passed + observation['expansions'][agent.get_id()]*self.T
                # Check whether current agent is not terminated
                if agent.is_agent_terminated(self.agents_times[agent.get_id()] > self.deadline):
                    actions.append(None)
                    continue
                actions.append(action)
                
                # Apply action and receive the last observation
                observation = env.apply_action(action)
                if observation['collected']:
                    self.agents_pickup_times[agent.get_id()] = self.agents_times[agent.get_id()]
                if action['action_tag'] == 'terminate':
                    agent.is_agent_terminated(True)
            self.print_env(observation,actions)
            if self.is_simulation_finished(agent_list,observation):
                # is_finished = True
                break
        results = [(self.agents_pickup_times[agent.get_id()],env.people_collected[a.get_id()],a.type) for a in agent_list]
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
        print('\n\n\n')
        if init:
            print("================Initial Enviorment===============")
        else:
            print(f"\n=====Time-step is {self.time_passed}/{self.deadline}=====\n")
        print("\n=====Graph information=====\n")
        G = observation['graph']
        P = observation['people_location']
        A = observation['agents_location']
        B = observation['blocked_edges']
        PC = observation['people_collected']
        E = observation['expansions']
        for n in G.graph:
            print(f'Node {n}:\n')
            print(f'\tPeople count is {P[n] if n in P else 0}')
            neighbors = [f"{n_tag}-{w}" for n_tag,w in G.get_neigbours_and_weights(n) if (n,n_tag) not in B]
            print(f'\tNeighbors-Distance: {", ".join(neighbors) if len(neighbors) else "No available neighbors!"}')
        print("\n=====Agents information=====\n")
        for a in A:
            term = not init and actions[a] is None
            n1,n2,d = A[a]
            if term:
                if n1 == n2:
                    print(f'(Terminated) Agent {a} collected {PC[a]} people, expanded {E[a]} nodes, current agent time is {self.agents_times[a]} and is at Node {n1} ')
                else:
                    print(f'(Terminated) Agent {a} collected {PC[a]} people, expanded {E[a]} nodes, current agent time is {self.agents_times[a]} and was crossing edge ({n1},{n2}) and completed {G.get_weight(n1,n2)-d}/{G.get_weight(n1,n2)} of traversal')
                continue
            action = actions[a]['action_tag'].upper() if a < len(actions) else None
            blocked = actions[a]['action_details']['to_block'] if action == 'BLOCK' else ''
            
            if n1 == n2:
                print(f'(Active) Agent {a} collected {PC[a]} people, expanded {E[a]} nodes, current agent time is {self.agents_times[a]} and is at Node {n1}{f" - last action: {action} {blocked}" if action else ""}')
            else:
                print(f'(Active) Agent {a} collected {PC[a]} people, expanded {E[a]} nodes, current agent time is {self.agents_times[a]} and is crossing edge ({n1},{n2}) and completed {G.get_weight(n1,n2)-d}/{G.get_weight(n1,n2)} of traversal')
        print('\n\n\n')


if __name__ == '__main__':
    Ts = [0,0.000001,0.01]
    results = []
    print('\n\nStarting HW2 Simulator!\n\n')
    while True:
        try:
            ans = input("Choose T:\n%s\n(%s) All of them\n\nPlease enter your choice: "%('\n'.join([f'({i}) {t}'for i,t in enumerate(Ts)]),len(Ts)))
            ans = int(ans)

            if ans != len(Ts) and ans >= 0:
                Ts = [Ts[ans]]
            break
        except:
            print('Must choose an integer in range')
      
    for T in Ts:
        print(f'Creating new simulator with T={T}')
        simulator = Simulator(T=T)
        agents, env = simulator.initialize_env()
        results.append(simulator.simulation_loop(env, agents))

    print('Preformance evaluation')
    
    for j,res in enumerate(results):
        print(f'\nFor T={Ts[j]}')
        for i,agent in zip(range(1,len(results)+1),res):
            time,ppl,t = agent
            print(f'Agent {i} of type {t}: collected {ppl} people in {time} time units and got a preformance measure of {ppl+ppl/time} which is number of people saved per time unit + the total amount of people collected by the agent') 
