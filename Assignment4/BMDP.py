from typing import List, Dict, Tuple
import json
from random import random
import itertools
import utils
from functools import reduce
import pprint

global_nodes_types_dict = {"vertex": {}, "edge_0": {}}  # type: Dict[str,Dict[str,Node]]
spontaneous_block_prob = 0.001

global_nodes = []
global_max_time = 0
global_number_of_iterations = 100000
start_node = None
target_node = None
global_edges = {}
global_truth = None
input_file = "graph1.json"


class BeliefState:
    def __init__(self, vertex, belief, relevant):
        self.vertex = vertex
        self.belief = belief
        self.neighbors = []
        self.reward = 1 if vertex == target_node else 0
        self.value = self.reward
        self.next = None
        self.relevant_beliefs = relevant if vertex != target_node else []
        self.relevant = []
        self.cost = 0

    def __str__(self):
        v_string = f'V{self.vertex}'
        e_string = [f'P(Blocked E{e})={self.belief[e]}' for e in self.belief]
        # print(self.value,self.cost)
        value_string = '\t\t\t\tV(s)=%f'%(self.value+self.cost)
        action_string = f'\t\t\t\tOptimal action: {"traverse to V" if self.next is not None else "terminate"}{self.next.vertex if self.next is not None else ""}'
        state_string = f'Belief state -  s=({v_string}, {", ".join(e_string)})\n{value_string}\n{action_string}'
        return state_string

    def add_neighbors(self, states, truth):
        if len(self.neighbors) != 0 or self.vertex == target_node:
            return
        for e in global_edges:
            if self.vertex not in e: continue
            n1, n2 = e
            n = n2 if n1 == self.vertex else n1
            possible_neighbors = [bs for bs in states if
                                  bs.vertex == n and (
                                          e not in self.belief or self.belief[e] != 1) and possible_next_state(self,
                                                                                                               bs,
                                                                                                               truth)]

            self.neighbors.extend(possible_neighbors)
        for n in self.neighbors:
            n.add_neighbors(states, truth)
        self.relevant = [bs for bs in states if bs.vertex == self.vertex and bs.belief in self.relevant_beliefs]
        for rs in self.relevant:
            rs.add_neighbors(states, truth)

    def compute_value(self, beliefs, seen, prev_cost):
        seen.add(self)
        values = dict()
        if self.vertex == target_node:
            return self.reward + prev_cost
        for n in self.neighbors:
            edge = frozenset([self.vertex, n.vertex])
            if n in seen:
                continue
            val = n.compute_value(beliefs, seen.copy(), -global_edges[edge][0])
            if edge in self.belief and self.belief[edge] == 1:
                val += prev_cost

            values[n] = val
        best_val = float('-inf')
        next_state = None
        for n in values:
            tmp_val = values[n]
            if tmp_val > best_val or next_state is None:
                best_val = tmp_val
                next_state = n
        self.value = best_val + prev_cost + self.reward
        self.next = next_state
        for n in values:
            tmp_val = values[n]
            edge = frozenset([self.vertex, n.vertex])
            if self.value - 2*global_edges[edge][0] > tmp_val:
                n.value = self.value - 2*global_edges[edge][0]
                n.next = self
        return self.value


    def reset_belief(self):
        self.neighbors = []
        self.value = self.reward
        self.next = None
        self.cost = 0


def possible_next_state(prev_state: BeliefState, state: BeliefState, truth):
    curr_b = state.belief
    prev_b = prev_state.belief
    # print(truth)
    count = 0
    for e in curr_b:
        if curr_b[e] != prev_b[e] and (0 == prev_b[e] or prev_b[e] == 1 or (
                (curr_b[e] == 0 or curr_b[e] == 1) and (curr_b[e] != truth[e]) or state.vertex not in e)):
            return False
    return True


class BeliefStateMDP:
    def __init__(self, state: BeliefState, states: List[BeliefState], truth, beliefs):
        self.initial = state
        self.truth = truth
        state.add_neighbors(states, truth)

        for _ in range(1):
            values = []
            vals = []
            probs = []
            next_state = None
            for rs in state.relevant:
                val = rs.compute_value(beliefs, set(), 0)
                prob = reduce(lambda a, b: a * b, [beliefs[e] if rs.belief[e] == 1 else (1 - beliefs[e]) for e in rs.belief], 1)
                vals.append(val)
                probs.append(prob)
                values.append(round(val*prob,3))
                if val*prob > max(values):
                    next_state = rs.next
            state.value = sum(values)
            for n in state.neighbors:
                if n.vertex == next_state.vertex:
                    state.next = n


def load_file(file_name):
    global global_nodes, global_edges, start_node, target_node
    with open(file_name) as f:
        json_dict = json.load(f)
    global_nodes = list(range(1, json_dict["nodes_num"] + 1))
    start_node = json_dict["start"]
    target_node = json_dict["target"]
    global_edges = {}
    for e in json_dict["edges"]:
        edge = frozenset(e[:2])
        weight = e[2]
        prob = 0 if len(e) < 4 else e[3]
        global_edges[edge] = [weight, prob]
    direct_edge = frozenset({start_node, target_node})
    if direct_edge not in global_edges:
        global_edges[direct_edge] = [10000, 0]
    states = []
    beliefs = {e: global_edges[e][1] for e in global_edges if global_edges[e][1] > 0}
    b_prod = [dict(x) for x in list(itertools.product(*[[(e, 0), (e, 1), (e, beliefs[e])] for e in beliefs]))]
    for n in global_nodes:
        for belief in b_prod:
            tmp = [b for b in belief if 0 == belief[b] or belief[b] == 1 or n not in b]
            if len(tmp) != len(belief): continue
            relevant = [bb for bb in b_prod if bb != belief and all(
                [(0 < belief[e] < 1 and bb[e] != belief[e]) for
                 e in belief])]
            states.append(BeliefState(n, belief, relevant))
    return states, beliefs


def test(bs, beliefs, truth):
    test = [bs.belief[e] == beliefs[e] if (start_node not in e) or (e not in truth) else bs.belief[e] != truth[e] for e
            in bs.belief]
    return test


def generate_instance(states, beliefs):
    truth = {e: 1 if random() < beliefs[e] else 0 for e in beliefs}
    for state in states:
        state.reset_belief()
    possible_initial = [bs for bs in states if bs.vertex == start_node and all(test(bs, beliefs, truth))]
    if len(possible_initial) != 1:
        print('WTF WTF!!!!')
        exit(0)
    mdp = BeliefStateMDP(possible_initial[0], states, truth, beliefs)
    return mdp


def print_graph(mdp):
    if mdp is None:
        print('No instance available!')
        return
    print(f'Number of vertices: {len(global_nodes)}')
    print(f'Number of edges: {len(global_edges)}')
    print('Edges weights:')
    for e in global_edges:
        print(f'w(E{e}) = {global_edges[e][0] if e not in mdp.truth or mdp.truth[e] == 0 else "inf"}')


def states2strings(states):
    ans = set()
    for bs in states:
        v_string = f'V{bs["v"]}'
        e_string = [f'P(Blocked E{e})={bs["belief"][e]}' for e in bs['belief']]
        ans.add(f'Belief state - s=({v_string}, {", ".join(e_string)})')
    return ans


def print_all_states(seen, state: BeliefState):
    seen.add(state)
    for bs in state.neighbors:
        if bs in seen: continue
        print(bs)
        seen.add(bs)
        seen = print_all_states(seen, bs)
    return seen


def print_mdp(mdp, states):
    if mdp is None:
        print('No instance available!')
        return
    print('The belief state MDP of the instance:')
    print(mdp.initial)
    seen = print_all_states(set(), mdp.initial)
    for bs in states:
        if bs not in seen:
            k = str(bs).split("\n")[0]
            print(f'{k} - is unreachable')


def run_policy(mdp):
    current = mdp.initial.vertex
    next_state = mdp.initial.next
    seq = f'{current}'
    print(f'Expected cost: {mdp.initial.value}')
    cost = 1
    while next_state is not None:
        cost -= global_edges[frozenset([current, next_state.vertex])][0]
        current = next_state.vertex
        next_state = next_state.next
        seq += f'->{current}'
    print(seq+'->Terminate')
    print(f'Actual cost: {cost}')


def main_menu(states, beliefs):
    q = False
    mdp = None
    while q != True:
        res = utils.promptMenu("\n\nChoose an action\n",
                               {"Generate random instance": (lambda: generate_instance(states, beliefs)),
                                "Display Graph": lambda: print_graph(mdp),
                                "Display Belief states": (lambda: print_mdp(mdp, states)),
                                "Display policy + run on graph": lambda: run_policy(mdp),
                                "Quit": lambda: True})
        q = res()
        if isinstance(q, BeliefStateMDP):
            mdp = q
            print('Generated new instance!')


if __name__ == '__main__':
    s, b = load_file(input_file)
    main_menu(s, b)
