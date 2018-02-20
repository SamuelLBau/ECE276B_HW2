import numpy as np
import Queue as Q

inf = float("inf")

class OPEN_list(Q.PriorityQueue):
    def __init__(self):
        Q.PriorityQueue.__init__(self) 
        self.iters = 0
    def get(self,ignore=False):
        if not ignore:
            self.iters += 1
        return Q.PriorityQueue.get(self)
    def peek(self):
        data = self.get(ignore=True)
        Q.PriorityQueue.put(self,data)
        return data
    def get_iter(self):
        return self.iters
def heuristic(nodeA,nodeB,coord_data,e=0):
    pos_A = coord_data[nodeA]
    pos_B = coord_data[nodeB]
    return e * np.linalg.norm(pos_A-pos_B)
def DJI_shortest_path(weight_data,coord_data):
    return shortest_path(weight_data,coord_data,0)
def AS_shortest_path(weight_data,coord_data):
    return shortest_path(weight_data,coord_data,1)
def WAS_shortest_path(weight_data,coord_data,epsilon=[1]):
    cost_list = []
    iter_list = []
    for e in epsilon:        
        [cost,iter] = shortest_path(weight_data,coord_data,e)
        cost_list.append(cost)
        iter_list.append(iter)
    
    print("WAS %s"%(str([cost_list,iter_list])))
    return [cost_list,iter_list]
def shortest_path(weight_data,coord_data,e):
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
    
    h_list = np.zeros([num_nodes])
    for i in range(num_nodes):
        h_list[i] = heuristic(i,last_node,coord_data,e)
    Pq.put([h_list[first_node],first_node])    
        
    closed_list = []
    while not Pq.empty():
        #This is used as a step to check if a node should be ignored
        #This is used in place as of get, as it is an implementation detail, not a performance detail
        temp_node = Pq.peek()[1]
        if (temp_node in closed_list):
            Pq.get(ignore=True)
            continue
        cur_dat = Pq.get()
        cur_node = cur_dat[1]
        closed_list.append(cur_node)
        cur_weight = weight_list[cur_node]
        if cur_node == last_node:
            break
        for i in range(num_nodes):
            new_weight = cur_weight + weights[cur_node,i]
            if new_weight < weight_list[i]:
                if new_weight + h_list[i] < weight_list[last_node]:
                    weight_list[i] = new_weight 
                    Pq.put([new_weight + h_list[i],i])
    
    cost = weight_list[last_node]
    iter = Pq.get_iter()
    #print("DJI %s"%(str([cost,iter])))
    return [cost,iter]