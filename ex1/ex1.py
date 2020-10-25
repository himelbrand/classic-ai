import re

class Vertex:
    def __init__(self,idx,p):
        self.idx = idx
        self.p = p
        self.neighbors = {} #each neighbor is a tuple of the form (weight,blocked)

    def add_neighbor(self,neighbor,weight,blocked=False):
        self.neighbors[neighbor] = (weight,blocked)

    def save_persons(self):
        p = self.p
        self.p = 0
        return p

    def block_road(self,neighbor):
        self.neighbors[neighbor][1] = True
        neighbor.block_road(self)

    def unblock_road(self,neighbor):
        self.neighbors[neighbor][1] = False
        neighbor.unblock_road(self)

class Agent:
    def __init__(self,start_pos,agent_type,state={}):
        self.state = state
        self.type = agent_type
        self.state['position'] = start_pos
        self.state['time_active'] = 0

    def traverse(self):
        pass
    
    def terminate(self):
        pass

class Enviorment:
    def __init__(self,confpath):
        self.parse_config(confpath)
        self.query_user()
        self.agents = []
        self.scores = []

    def parse_config(self,f):
        with open(f) as data:
            lines = data.readlines()
            lines = [line.split(';')[0].strip().split() for line in lines]
            for l in lines:
                if l[0] == 'N':
                    self.N = int(l[1])
                    self.vertices = [None for i in range(self.N+1)]
                if l[0] == 'D':
                    self.deadline = int(l[1])
                elif re.search(r'V\d+',l[0]):
                    idx = int(l[0][1:]) 
                    p = int(l[1]) if len(l) > 1 else 0
                    self.vertices[idx] = Vertex(idx,p)
                elif re.search(r'E\d+',l[0]):
                    i1,i2,w = [int(x) for x in l[1:]]
                    v1,v2 = self.vertices[i1],self.vertices[i2]
                    v1.add_neighbor(v2,w)
                    v2.add_neighbor(v1,w)

    def query_user(self):
        print('Please enter initial parameters...')
        agents_num = 0
        while not agents_num:
            try:
                agents_num = int(input('How many active agents ?'))
            except:
                print('Value must be a positive integer!')
        print('Possible agent types are: \nH or h - Human agent\nG1 or g1 - Part1 greedy agent\nS or s - Saboteur agent')
        for i in range(agents_num):
            agent_type = input('Enter agent type for agent %d'%i).lower()
            try:
                start_pos = int(input('What vertex number is the starting position for this agent?'))
                start_v = self.vertices[start_pos]
            except:
                print('Value must be a positive integer! and lower or equal to %d'%self.N)

class Simulator:
    def __init__(self):
        pass