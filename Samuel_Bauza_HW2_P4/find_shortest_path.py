import numpy as np
import Queue as Q

inf = float("inf")

class OPEN_list(Q.PriorityQueue):
    def __init__(self):
        Q.PriorityQueue.__init__(self) 
        self.iters = 0
    def get(self):
        self.iters += 1
        return Q.PriorityQueue.get(self)
    def get_iter(self):
        return self.iters
def DJI_heuristic(nodeA,nodeB,coord_data):
    return 0
def AS_heuristic(nodeA,nodeB,coord_data):
    pos_A = coord_data[nodeA]
    pos_B = coord_data[nodeB]
    return np.linalg.norm(pos_A-pos_B)
def WAS_heuristic(nodeA,nodeB,coord_data,e=2):
    return e * AS_heuristic(nodeA,nodeB,coord_data)
def DJI_shortest_path(weight_data,coord_data):
    return AS_shortest_path(weight_data,coord_data,DJI_heuristic)
def AS_shortest_path(weight_data,coord_data,heuristic=AS_heuristic):
    cost    = 0
    iter   = 0
    weights = weight_data[0]
    num_nodes = len(weights)
    [first_node,last_node] = [weight_data[1],weight_data[2]]
    num_nodes = len(weights)
    #Begin Djikstra shortest path algorithm
    Pq = OPEN_list()
    weight_list = np.ones(weights.shape[0])*inf
    weight_list[first_node] = 0
    Pq.put([0,first_node])
    
    h_list = np.zeros([num_nodes])
    for i in range(num_nodes):
        h_list[i] = heuristic(i,last_node,coord_data)
    
    while not Pq.empty():
        cur_dat = Pq.get()
        cur_node = cur_dat[1]
        cur_weight = cur_dat[0]
        for i in range(num_nodes):
            new_weight = cur_weight + weights[cur_node,i]
            if new_weight < weight_list[i]:
                if new_weight + h_list[i] < weight_list[i]:
                    weight_list[i] = new_weight
                    Pq.put([new_weight,i])
    
    cost = weight_list[last_node]
    iter = Pq.get_iter()
    #print("DJI %s"%(str([cost,iter])))
    return [cost,iter]
def WAS_shortest_path(weight_data,coord_data,epsilon=[1]):
    cost_list = []
    iter_list = []
    for e in epsilon:        
        weights = weight_data[0]
        num_nodes = len(weights)
        [first_node,last_node] = [weight_data[1],weight_data[2]]
        num_nodes = len(weights)
        #Begin Djikstra shortest path algorithm
        Pq = OPEN_list()
        weight_list = np.ones(weights.shape[0])*inf
        weight_list[first_node] = 0
        
        h_list = np.zeros([num_nodes])
        for i in range(num_nodes):
            h_list[i] = WAS_heuristic(i,last_node,coord_data,e=e)
        
        Pq.put([h_list[first_node],first_node])
        closed = {}
        while not last_node in closed:
            #print("AGAIN")
            cur_dat = Pq.get()
            cur_node = cur_dat[1]   #i
            cur_weight = cur_dat[0]
            closed[cur_node]=True
            for j in range(num_nodes):
                if j in closed:
                    continue
                new_weight = cur_weight + weights[cur_node,j]
                #print(weights[cur_node,j])
                if new_weight < weight_list[j]:
                    weight_list[j] = new_weight
                    Pq.put([new_weight + h_list[j],j])
        cost = weight_list[last_node]
        iter = Pq.get_iter()
        cost_list.append(cost)
        iter_list.append(iter)
    
    print("WAS %s"%(str([cost_list,iter_list])))
    return [cost_list,iter_list]