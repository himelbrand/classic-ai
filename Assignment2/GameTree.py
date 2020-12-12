from distance_algs import get_shortest_path_Dijk

class AgentState:
    def __init__(self,source,dest,score,terminated,time,distance,last_visit,agent_id):
        self.source = source
        self.dest = dest
        self.score = score
        self.distance = distance
        self.terminated = terminated
        self.time = time
        self.last_visit = last_visit
        self.id = agent_id
    def keep_traversing(self,current=False):
        d = max(0,self.distance - 1) if current else self.distance
        return AgentState(self.source if d else self.dest,self.dest,self.score,self.terminated,self.time+1,d,self.last_visit,self.id)
    def terminate(self,reward):
        return AgentState(self.source,self.dest,self.score+reward,True,self.time+1,0,self.last_visit,self.id)
    def traverse(self,reward,dest,distance):
        return AgentState(self.source,dest,self.score+reward,self.terminated,self.time+1,distance-1,self.source,self.id)
    def __str__(self):
        return f'{self.source}-{self.score}-{self.dest}-{self.distance}-{self.terminated}'


class GameState:
    def __init__(self,agent1:AgentState,agent2:AgentState,people_locs,edges,graph,deadline):
        self.agent1 = agent1
        self.agent2 = agent2
        self.people_locs = people_locs.copy()
        self.edges = edges
        self.graph = graph
        self.deadline = deadline
    
    def successor(self,go_to):
        reward = self.people_locs[self.agent1.source]
        people_locs = self.people_locs.copy()
        people_locs[self.agent1.source] = 0
        e = (go_to,self.agent1.source) if go_to < self.agent1.source else (self.agent1.source,go_to)
        agent = self.agent1.traverse(reward,go_to,self.edges[e]) if self.agent1.source != go_to else self.agent1.terminate(reward)
        return GameState(self.agent2.keep_traversing(),agent,people_locs.copy(),self.edges,self.graph,self.deadline)
    
    def successors(self):
        if self.agent1.time >= self.deadline:
            self.agent1.terminated = True
        if self.agent1.terminated and self.agent2.terminated:
            return []
        if self.agent1.terminated:
            return [(('terminate',self.agent1.dest),GameState(self.agent2.keep_traversing(),self.agent1.keep_traversing(current=True),self.people_locs.copy(),self.edges,self.graph,self.deadline))]
        if self.agent1.distance: 
            return [(('traverse',self.agent1.dest),GameState(self.agent2.keep_traversing(),self.agent1.keep_traversing(current=True),self.people_locs.copy(),self.edges,self.graph,self.deadline))]
        else:
            return  [(('terminate',self.agent1.source),self.successor(self.agent1.source))] + [(('traverse',dest),self.successor(dest)) for dest in self.graph[self.agent1.source]]
    def isTerminal(self):
        return (self.agent1.terminated and self.agent2.terminated)  or sum(self.people_locs)==0
    def __str__(self):
        a1,a2 = (self.agent1,self.agent2) if self.agent1.id < self.agent2.id else (self.agent2,self.agent1)
        return f'{self.people_locs}_{a1}_{a2}'
        
class Node:
    def __init__(self,state:GameState,current_level:int,cutoff:int,visited:set):
        if current_level%2 == 0:
            self.scores = (state.agent1.score,state.agent2.score)
        else:
            self.scores = (state.agent2.score,state.agent1.score)
        self.state = state
        self.level = current_level
        self.successors = [] if self.level == cutoff else [(a,Node(s,self.level+1,cutoff,visited)) for a,s in state.successors() if str(s) not in visited]
    
    def isTerminal(self):
        return len(self.successors) == 0

    def __str__(self):
        return str(self.state)   

def basic_h(node:Node,G):
        state = node.state
        agent1 = state.agent1 if node.level%2==0 else state.agent2
        agent2 = state.agent2 if node.level%2==0 else state.agent1
        s1,s2 = 0,0
        p1 = int(node.level%2)
        p2 = int(node.level%2==0)
        for dest in state.people_locs:
            ppl = state.people_locs[dest]
            if ppl == 0: continue
            d1,_ = get_shortest_path_Dijk(G,agent1.source,dest) if not agent1.terminated else (float('inf'),0)
            d2,_ = get_shortest_path_Dijk(G,agent2.source,dest) if not agent2.terminated else (float('inf'),0)
            d1*=2
            d2*=2
            d1 += p1
            d2 += p2
            
            if d1 < d2:
                s1 += (ppl-d1*0.001) if d1 and d1 + agent1.time < state.deadline else ppl if d1 + agent1.time < state.deadline else 0
            else:
                s2 += (ppl-d2*0.001) if d2 and d2 + agent2.time < state.deadline else ppl if d2 + agent2.time < state.deadline else 0
        t = agent1.time*0.0001
        return (s1-t,s2-t)



class GameTree:
    @classmethod 
    def generateGameState(cls,agent1,agent2,observation):
        deadline = observation['deadline']
        graph = observation['graph']
        o1,t1,d1 = observation['agents_location'][agent1.get_id()]
        o2,t2,d2 = observation['agents_location'][agent2.get_id()]
        A1 = AgentState(o1,t1,observation['people_collected'][agent1.get_id()],agent1.is_terminated,agent1.t,d1,agent1.location,agent1.get_id())
        A2 = AgentState(o2,t2,observation['people_collected'][agent2.get_id()],agent2.is_terminated,agent2.t,d2,agent2.location,agent2.get_id())
        return GameState(A1,A2,observation['people_location'].copy(),graph.weights,graph.graph,deadline)

    def __init__(self,agent1,agent2,observation,cutoff,h=basic_h):
        self.h = h
        self.deadline = observation['deadline']
        self.graph = observation['graph']
        self.cutoff = cutoff
        o1,t1,d1 = observation['agents_location'][agent1.get_id()]
        o2,t2,d2 = observation['agents_location'][agent2.get_id()]
        A1 = AgentState(o1,t1,observation['people_collected'][agent1.get_id()],agent1.is_terminated,agent1.t,d1,agent1.location,agent1.get_id())
        A2 = AgentState(o2,t2,observation['people_collected'][agent2.get_id()],agent2.is_terminated,agent2.t,d2,agent2.location,agent2.get_id())
        self.state = GameState(A1,A2,observation['people_location'].copy(),self.graph.weights,self.graph.graph,self.deadline)
    

    def maxValue(self,state,alpha,beta,utility,minFoo):
        if state.isTerminal():
            return utility(state)
        s_successors = state.successors
        v = float('-inf')
        scores = (float('-inf'),float('-inf'))
        for _,s in s_successors:
            v_tmp,scores_tmp = minFoo(s,alpha,beta,utility,self.maxValue)
            if v < v_tmp:
                v = v_tmp
                scores = scores_tmp
            if v >= beta: return v,scores
            alpha = max(alpha,v)
        return v,scores

    def minValue(self,state,alpha,beta,utility,maxFoo):
        if state.isTerminal():
            return utility(state)
        s_successors = state.successors
        v = float('inf')
        scores = (float('inf'),float('inf'))
        for _,s in s_successors:
            v_tmp,scores_tmp = maxFoo(s,alpha,beta,utility,self.minValue)
            if v > v_tmp:
                v = v_tmp
                scores = scores_tmp
            if v <= alpha: return v,scores
            beta = min(beta,v)
        return v,scores

    def zero_sum_utility(self,node:Node):
        def diff(x):
            return x[0]-x[1],x
        state = node.state
        g,gscore = diff(node.scores)
        if state.isTerminal(): return g,gscore
        h = self.h(node,self.graph)
        f,fscore = diff((gscore[0]+h[0],gscore[1]+h[1]))
        return f,fscore

    def alpha_beta_decision(self,visited):
        root = Node(self.state,0,self.cutoff,visited)
        best_a = None
        max_val = float('-inf')
        for a,s in root.successors:
            
            v,_ = self.minValue(s,float('-inf'),float('inf'),self.zero_sum_utility,self.maxValue)
            if v > max_val:
                max_val = v
                best_a = a
        return best_a

    def cooperative_utility(self,state):
        h = self.h(state,self.graph)
        g = state.scores
        fscore = (h[0]+g[0],g[1]+h[1])
        return fscore
    
    def maxSemi(self,state,i,utility,visited):
        scores = (float('-inf'),float('-inf'))
        if state.isTerminal():
            return utility(state)
        s_successors = state.successors
        
        for _,s in s_successors:
            if str(s) in visited:
                continue
            i = s.level%2
            j = (i+1)%2
            scores_tmp = self.maxSemi(s,(i+1)%2,utility,visited)
            if scores_tmp[j] > scores[j]:
                scores = scores_tmp
            elif scores_tmp[j] == scores[j] and scores_tmp[i] > scores[i]:
                scores = scores_tmp
        return scores

    def semi_decision(self,visited):
        root = Node(self.state,0,self.cutoff,visited)
        best_a = None
        scores = (float('-inf'),float('-inf'))
        for a,s in root.successors:
            scores_tmp = self.maxSemi(s,0,self.cooperative_utility,visited)
            if scores_tmp[0] > scores[0]:
                scores = scores_tmp
                best_a = a
            elif scores_tmp[0] == scores[0] and scores_tmp[1] > scores[1]:
                scores = scores_tmp
                best_a = a
        return best_a

    def maxFully(self,state,utility,visited):
        if state.isTerminal():
            return sum(utility(state)) 
        s_successors = state.successors
        v = float('-inf')
        for _,s in s_successors:
            v_tmp = self.maxFully(s,utility,visited) 
            if v_tmp > v:
                v = v_tmp
        return v

    def fully_decision(self,visited):
        root = Node(self.state,0,self.cutoff,visited)
        best_a = None
        v_max = float('-inf')
        for a,s in root.successors:
            v = self.maxFully(s,self.cooperative_utility,visited)
            if v > v_max:
                v_max = v
                best_a = a
        return best_a
