import Env1
import Agent
from typing import List

global_file_name = "graph2.json"


class Simulator:
    agent_init_functions = {"human": Agent.HumanAgent.create_agent,
                            "greedy": Agent.GreedyAgent.create_agent,
                            "sabateur": Agent.SaboteurAgent.create_agent,
                            "planning":Agent.PlanningAgent.create_agent}  # Agent type and the function that creates particular agent
    deadline = 0

    def __init__(self):
        self.time_passed = 0

    def initialize_env(self):
        """Environment initialization and agents creating function """

        # Environment source file : should be json file
        global global_file_name

        # Agents creating
        print("How many agents do you want to have ?")
        agent_num = int(input())

        agents_list = []  # type: List [Agent]
        agent_locations = {}

        for i in range(1, agent_num + 1):
            agent_types = list(Simulator.agent_init_functions.keys())
            print("Agent N_{} : Please enter the agent type (one of the following {}) ?".format(i, agent_types))
            agent_type = input()

            print("Please enter agent location ( node number ) ")
            agent_location = int(input())

            # Invoking agent initialization function ( according to agent type )
            agent = Simulator.agent_init_functions[agent_type]()
            agents_list.append(agent)
            agent_locations[agent.get_id()] = [agent_location, agent_location, 0]

        print("Reading the environment input file...")

        graph, people, Simulator.deadline = Env1.load_environment(global_file_name)

        print("Initializing environment")

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

        # TODO Add function that checks whether the simulation is finished
        # TODO Add the func that print the world state
        # The main loop
        while not is_finished:
            for agent in agent_list:
                # Check whether current agent is not terminated
                if agent.is_agent_terminated():
                    continue
                # Recieve action from the current agent
                action = agent.next_action(observation)

                self.time_passed += 1

                # Apply action and receive the last observation
                observation = env.apply_action(action)

                if self.is_simulation_finished(agent_list,observation):
                    is_finished = True
                    break

    def print_statistics(self):
        ...

    def record_statistics(self, observation):
        ...

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


if __name__ == '__main__':
    simulator = Simulator()

    agents, env = simulator.initialize_env()

    simulator.simulation_loop(env, agents)
