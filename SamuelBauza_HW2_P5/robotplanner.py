import numpy as np
import scipy as sci
import Queue as Q
import math
import time
from copy import deepcopy

import sys

MINIMUM_DENSITY = 100    #Average minimum per NxN space
MAX_SAMPLES     = 0         #NOTE: robot_pos and target_pos will always be included, as will samples generated during processing
NEAREST_NEIGHBORS_N = 20
MAX_A_DIST      = 40
MAX_A_STEPS     = int(math.ceil(MAX_A_DIST * 1.5))
first_iter = True
debug = False
debug_prompt = True

robot_pos_prev = 0
target_pos_prev = 0

sparse_map = None
OBSTACLE_VALUE = 1
FREE_VALUE      =0
current_weight_for_debug = 0

def get_node_distance(env_map,nodeA,nodeB):
    if isinstance(nodeA,int):
        nodeA = num_to_index(env_map,nodeA)
    if isinstance(nodeB,int):
        nodeB = num_to_index(env_map,nodeB)
    return abs(nodeA[0]-nodeB[0]) + abs(nodeA[1]-nodeB[1])    
def get_adjacent_nodes(env_map,nodeA):
    #This is specifically for environment map
    intmode = not isinstance(nodeA,np.ndarray)
    if intmode:
        nodeA = num_to_index(env_map,nodeA)
    valid_moves = []
    map_shape = env_map.shape
    for y in range(nodeA[0]-1,nodeA[0]+2):
        if( y < 0 or y >= map_shape[0]):
            continue
        for x in range(nodeA[1]-1,nodeA[1]+2):
            if( x < 0 or x >= map_shape[1]):
                continue
            if env_map[y,x] == FREE_VALUE:
                if intmode:
                    valid_moves.append(index_to_num(env_map,np.array([y,x])))
                else:
                    valid_moves.append([y,x])
    return valid_moves  
def attempt_connect_components(env_map,node_list,compA,compB):
    #Returns true if it manages to connect components, otherwise returns false
    
    '''
        Randomly pick point in both compA and compB
        
        
        flood from pointA until pointB reached, keeping track
    '''
    pointA = compA[0]
    pointB = compB[0]
    
    weight_map = np.ones(env_map.size)*float("inf")
    weight_map[pointA] = 0
    
    cur_queue = Q.Queue()
    cur_queue.put(pointA)
    
    done = False
    found = False
    cur_weight = 1
    flood_val = 0
    if debug:
        env_map_copy = deepcopy(env_map)
        pointA_i     = num_to_index(env_map,pointA)
        pointB_i     = num_to_index(env_map,pointB)
    while not done and not found:
        sys.stdout.write("Flooding dist: %d\r"%(flood_val))
        flood_val+=1
        sys.stdout.flush()
        done = True
        next_queue = Q.Queue()
        id=0
        while not cur_queue.empty():
            id+=1
            cur_node = cur_queue.get()
            adj_nodes = get_adjacent_nodes(env_map,cur_node)
            if debug:
                cur_node_i = num_to_index(env_map_copy,cur_node)
                env_map_copy[cur_node_i[0],cur_node_i[1]] = .5
            for node in adj_nodes:
                if weight_map[node] == float("inf"):
                    weight_map[node] = cur_weight
                    if not node == pointB:
                        done = False
                        next_queue.put(node)
                    else:
                        found=True
                        break
            if found:
                break
        cur_weight += 1
        cur_queue.queue = deepcopy(next_queue.queue)
        if debug and flood_val%400==0:
            display_map_with_nodes(env_map_copy,pointA_i,pointB_i,pause=False,delete=False)
    if found:
        prev_point = pointB             #Previous max dist point in train
        cur_weight = weight_map[pointB]
        cur_point = pointB              #Current point
        next_point = pointB
        
        connected = False
        while not connected:
            distAB = nodes_can_connect(env_map,prev_point,next_point)
            distBA = nodes_can_connect(env_map,next_point,prev_point)
            while (distAB > 0 and distBA > 0) or prev_point == next_point:
                cur_point = next_point
                adj_nodes = get_adjacent_nodes(env_map,cur_point)
                cur_weight = weight_map[cur_point]
                
                if cur_weight == 0:
                    break
                min_weight = cur_weight
                next_point = cur_point
                for node in adj_nodes:
                    if(min_weight > weight_map[node]):
                        min_weight = weight_map[node]
                        next_point = node
                if(np.all(cur_point==next_point)):
                    raise Exception("Error connecting points during expansion")
                distAB = nodes_can_connect(env_map,prev_point,next_point)
                distBA = nodes_can_connect(env_map,next_point,prev_point)
            add_node(env_map,node_list,cur_point)
            add_edge(env_map,node_list,cur_point,prev_point)
            if cur_point in compA:
                connected = True
            else:
                for point in node_list[cur_point]:
                    if point in compA:
                        add_edge(env_map,node_list,cur_point,point)
                        connected = True
                        break
            if not connected:
                prev_point = cur_point
    else:#Failed to connect these two components
        pass
    sys.stdout.write("\n")
    return found
def add_connected_nodes(key,node_list,connected_list):
    
    if not key in connected_list:
        connected_list.append(key)
        for item in node_list[key]:
            add_connected_nodes(item,node_list,connected_list)
def expand_map(env_map,node_list,robot_pos,target_pos):

    #Determine seperate connected components
    node_list_copy = deepcopy(node_list)
    disconnected_comp_list = []
    for key,item in node_list.items():
        for comp in disconnected_comp_list:
            if key in comp:
                break
        else:
            cur_connected = []
            add_connected_nodes(key,node_list,cur_connected)
            disconnected_comp_list.append(np.array(deepcopy(cur_connected)))
    for id,comp in enumerate(disconnected_comp_list):
        if robot_pos in comp:
            robot_id = id
        if target_pos in comp:
            target_id = id
    #temp = deepcopy(disconnected_comp_list[target_id])
    ##del disconnected_comp_list[target_id]
    #disconnected_comp_list.insert(0,temp)
    #if robot_id < target_id:
    #    robot_id += 1
    #temp = deepcopy(disconnected_comp_list[robot_id])
    #del disconnected_comp_list[robot_id]
    #disconnected_comp_list.insert(0,temp)
    
    #Connect disconnected components until all are connected
    is_valid = True
    print("Detected %d disconnected components. Attempting to connect robot and target components, this may take a few minutes on large maps. "%(len(disconnected_comp_list)))
    
    #NOTE: The logic is a little off here because I decided to only connect robot and target components
    #Most of the logic is available to connect all components, but this took significant initialization
    #time with little to no performance boost
    skip_list = []
    for t in range(1):#Only do first iteration
        cur_compA = disconnected_comp_list[robot_id]
        cur_compB = disconnected_comp_list[target_id]
        if attempt_connect_components(env_map,node_list,cur_compA,cur_compB):
            new_comp = np.concatenate([cur_compA,cur_compB])
            print("Connecting components %d and %d, %d disconnected components remain"%(\
                robot_id,target_id,len(disconnected_comp_list)-1))
            del disconnected_comp_list[robot_id]
            if target_id > robot_id:
                target_id -= 1
            del disconnected_comp_list[target_id]
            disconnected_comp_list.append(new_comp)
            print("Expansion successful")
            break
        else:
            print("Expansion not successful")
            is_valid = False
        if debug:
            display_map_with_nodes(env_map,num_to_index(env_map,robot_pos),num_to_index(env_map,target_pos),pause=False)
    return is_valid
def get_path_full(env_map,robot_pos,target_pos,max_dist = MAX_A_STEPS):
    #Returns next target node, not next position
    #Maximum depth is max_dist, this should be used to limit exploration time of A*
    #It is the maximum node depth A* should explore
    #returns -1 if no path found
    next_node    = -1
    num_nodes = env_map.size
    weights = np.ones([num_nodes])*float("inf")
    
    #Begin Djikstra shortest path algorithm
    robot_pos_num = index_to_num(env_map,robot_pos)
    target_pos_num = index_to_num(env_map,target_pos)
    Pq = Q.PriorityQueue()
    weights[target_pos_num] = 0
    Pq.put([0,target_pos_num])
    
    closed_list = []
    while not Pq.empty():
        cur_dat = Pq.get()
        cur_node = cur_dat[1]
        
        if cur_node in closed_list:
            continue
        closed_list.append(cur_node)
        cur_weight = weights[cur_node]
        new_nodes = get_adjacent_nodes(env_map,cur_node)
        for i in new_nodes:
            cur_i_i = num_to_index(env_map,i)
            new_weight = cur_weight + 1
            cur_dist = 0#get_node_distance(env_map,robot_pos,cur_i_i)
            if new_weight < weights[i] and new_weight <= MAX_A_STEPS:
                if new_weight + cur_dist < weights[robot_pos_num]:
                    weights[i] = new_weight
                    Pq.put([new_weight+cur_dist,i])
    
    adj_nodes = get_adjacent_nodes(env_map,robot_pos_num)
    cur_weight = weights[robot_pos_num]

    for node in adj_nodes:
        if weights[node] < cur_weight:
            weight_min = weights[node]
            next_node = node
            
    global current_weight_for_debug
    current_weight_for_debug = weight_min 
    return next_node
    
def get_path_sparse(env_map,nodelist,robot_pos,target_pos):
    #This uses A* and the generated sparse map to
    #This uses the sparse map to find a path from robot to target_pos
    #returns next target node, not next position
    #returns -1 if no path found
    
    next_node    = -1
    num_nodes = env_map.size
    weights = np.ones([num_nodes])*float("inf")
    
    #Begin Djikstra shortest path algorithm
    robot_pos_num = index_to_num(env_map,robot_pos) 
    target_pos_num = index_to_num(env_map,target_pos)
    Pq = Q.PriorityQueue()
    weights[target_pos_num] = 0
    Pq.put([0,target_pos_num])
    
    n_iter = 0
    closed_path = []
    while not Pq.empty():
        n_iter +=1
        cur_dat = Pq.get()
        cur_node = cur_dat[1]
        if cur_node in closed_path:
            continue
        if cur_node == robot_pos_num:
            break
        closed_path.append(cur_node)
        cur_weight = weights[cur_node]
        for i in nodelist[cur_node]:
            cur_i_i = num_to_index(env_map,i)
            new_weight = cur_weight + nodelist[cur_node][i]
            cur_dist = 0#get_node_distance(env_map,robot_pos,cur_i_i)
            if new_weight < weights[i]:
                if new_weight +cur_dist < weights[robot_pos_num]:
                    weights[i] = new_weight
                    Pq.put([new_weight+cur_dist,i])
    '''            
    print("START")
    print("robot_pos_num",robot_pos_num)
    print("target_pos_num",target_pos_num)
    print("WEIGHTS")
    
    for id,val in enumerate(weights):
        if not val == float("inf"):
            print(id,val)
    '''
    if weights[robot_pos_num] == float("inf"):
        return -1
    #print("N_ITER",n_iter)
    adj_nodes = nodelist[robot_pos_num]
    weight_min = weights[robot_pos_num]
    min_node = robot_pos_num
    #print("done printing")
    #raw_input()
    #print("node_weights",adj_nodes)
    for key in adj_nodes:
    #    print(key,weights[key],weights[robot_pos_num])
        if weight_min > weights[key]:
            weight_min = weights[key]
            min_node = key            
    #print("robot_pos_num",robot_pos_num)
    #print("min_node",min_node)
    global current_weight_for_debug
    current_weight_for_debug = weight_min
    return min_node

def nodes_can_connect(env_map,nodeA,nodeB,check_dist=False):    
    #returns 0 if nodes cannot connect
    if not isinstance(nodeA,np.ndarray):
        nodeA = num_to_index(env_map,nodeA)
    if not isinstance(nodeB,np.ndarray):
        nodeB = num_to_index(env_map,nodeB)
    node_distance = 0
    temp_dist = 0
    next_node = deepcopy(nodeA)
    while not np.all(next_node == nodeB):
        
        diff = nodeB - next_node
        
        if diff[0] > 0:
            diff[0] = 1
        elif diff[0] < 0:
            diff[0] = -1
        if diff[1] > 0:
            diff[1] = 1
        elif diff[1] < 0:
            diff[1] = -1
        next_node += diff
        if not env_map[next_node[0],next_node[1]] == FREE_VALUE:
            break
        temp_dist += 1
    else:
        node_distance = temp_dist
        
    if node_distance > 0:
        temp_dist = 0
        next_node = deepcopy(nodeB)
        node_distance = 0
        while not np.all(next_node == nodeA):
            
            diff = nodeA - next_node
            
            if diff[0] > 0:
                diff[0] = 1
            elif diff[0] < 0:
                diff[0] = -1
            if diff[1] > 0:
                diff[1] = 1
            elif diff[1] < 0:
                diff[1] = -1
            next_node += diff
            if not env_map[next_node[0],next_node[1]] == FREE_VALUE:
                break
            temp_dist += 1
        else:
            node_distance = temp_dist
    #print("node dist",node_distance)
    return node_distance
def peek_add_edges(env_map,nodelist,new_node,max_edges = NEAREST_NEIGHBORS_N):
    '''
        NOTE: THIS implements nearest N neighbors algorithm

    '''
    checked_neighbors = {}
    dist_queue = Q.PriorityQueue()
    for key in nodelist[new_node]:
        new_index = num_to_index(new_node)
        key_index = num_to_index(key)
        cur_dist = get_node_distance(env_map,new_index,key_index)
        dist_queue.put([cur_dist,key])
    
    edge_list = []
    max_edges = int(min(max_edges,9999999))
    for i in range(max_edges):
        if dist_queue.empty():
            break
        [dist,key] = dist_queue.get()
        edge_list.append([new_node,key,dist])
        #if not add_edge(env_map,nodelist,new_node,key):
        #    bad_edge_list.append([new_node,key])
    return edge_list
def add_edges(env_map,nodelist,new_node,max_edges = NEAREST_NEIGHBORS_N):
    '''
        NOTE: THIS implements nearest N neighbors algorithm

    '''
    checked_neighbors = {}
    dist_queue = Q.PriorityQueue()
    for key in nodelist:
        if key == new_node:
            continue
        new_index = num_to_index(env_map,new_node)
        key_index = num_to_index(env_map,key)
        cur_dist = get_node_distance(env_map,new_index,key_index)
        dist_queue.put([cur_dist,key])
        
    max_edges = int(min(max_edges,9999999))
    for i in range(max_edges):
        if dist_queue.empty():
            break
        [dist,key] = dist_queue.get()
        add_edge(env_map,nodelist,new_node,key)
        #if not add_edge(env_map,nodelist,new_node,key):
        #    bad_edge_list.append([new_node,key])
def add_edge(env_map,nodelist,new_node,test_node):
    if isinstance(new_node,int):
        new_node_i = num_to_index(env_map,new_node)
    else:
        new_node_i = new_node
    if isinstance(test_node,int):
        test_node_i = num_to_index(env_map,test_node)
    else:
        test_node_i = test_node
        
    ret_val=False
    dist = nodes_can_connect(env_map,new_node_i,test_node_i)
    if dist > 0:
        nodelist[test_node][new_node] = dist
        nodelist[new_node][test_node] = dist
        ret_val=True
    return ret_val
def add_node(env_map,nodelist,new_node,max_edges = NEAREST_NEIGHBORS_N):
    nodelist[new_node] = {}
    add_edges(env_map,nodelist,new_node,max_edges )
    
def initialize_sparse_map(env_map,robot_pos,target_pos):
    return_success = True
    global sparse_map
    num_nodes = env_map.size
    sparse_map = {}
    print("HERE")
    
    num_map_points = env_map.shape[0]*env_map.shape[1]
    num_free_samples = int(min(MAX_SAMPLES,math.ceil((num_map_points*1.0) / (MINIMUM_DENSITY*MINIMUM_DENSITY))))
    num_free_max = np.sum(env_map.astype(np.int))
    
    robot_pos_num = index_to_num(env_map,robot_pos)
    target_pos_num = index_to_num(env_map,target_pos)
    
    add_node(env_map,sparse_map,robot_pos_num)
    add_node(env_map,sparse_map,target_pos_num)
    #Randomly add nodes until minimum density achieved
    num_nodes = 2
    num = 0
    print("INITIALIZING %d points"%(num_free_samples))
    while num_nodes < num_free_samples:
        sys.stdout.write("%d of %d initialized\r"%(num_nodes,num_free_samples))
        sys.stdout.flush()
        num+=1
        id = np.random.randint(0,num_map_points-1)
        if not id in sparse_map:
            indices = num_to_index(env_map,id)
            if env_map[indices[0],indices[1]] == FREE_VALUE:
                add_node(env_map,sparse_map,id)
                num_nodes += 1
    display_map_with_nodes(env_map,robot_pos,target_pos,pause = debug_prompt)
    if get_path_sparse(env_map,sparse_map,robot_pos,target_pos) < 0:
        return_success =  expand_map(env_map,sparse_map,robot_pos_num,target_pos_num)
    return return_success
def initialize_planner(env_map,robot_pos,target_pos):
    global robot_pos_prev
    global target_pos_prev
    
    robot_pos_prev  = robot_pos
    target_pos_prev = target_pos    
    plan_success = initialize_sparse_map(env_map,robot_pos,target_pos)
    
    if not plan_success:
        raise Exception("Failed to find path")
    return plan_success
def index_to_num(env_map,index):
    return np.ravel_multi_index(index,env_map.shape)
def num_to_index(env_map,num):
    return np.array(np.unravel_index(num,env_map.shape))
def get_connected_nodes(node_list,start_node):
    con_list = []
    
    queue = Q.Queue()
    queue.put(start_node)
    while not queue.empty():
        cur_node = queue.get()
        con_list.append(cur_node)
        for node in node_list[cur_node]:
            if not node in con_list:
                queue.put(node)
    return con_list
def next_node_reccomendation(env_map,robot_pos,target_pos,nodelist):
    
    dist = get_node_distance(env_map,robot_pos,target_pos)
    new_robot_pos = robot_pos
    robot_pos_num = index_to_num(env_map,robot_pos)
    
    target_pos_num = index_to_num(env_map,target_pos)
    
    if not target_pos_num in nodelist:
        add_node(env_map,nodelist,target_pos_num)
        
        if not nodelist[target_pos_num]: #Checks if is necessary to flood find existing list
            connected_list = get_connected_nodes(nodelist,robot_pos_num)   
            attempt_connect_components(env_map,nodelist,[target_pos_num],connected_list)

    next_node_num = -1
    if dist < MAX_A_DIST:#If within a certain distance, attempt A*
        next_node_num = get_path_full(env_map,robot_pos,target_pos)
    if next_node_num < 0:
        next_node_num = get_path_sparse(env_map,nodelist,robot_pos,target_pos)
    if next_node_num < 0:
        raise Exception("NO PATH FOUND when searching")
    else:
        goal_node_num = int(next_node_num)
        next_node_calc = num_to_index(env_map,int(next_node_num))
        diff = next_node_calc - robot_pos
        if diff[0] > 0:
            diff[0] = 1
        elif diff[0] < 0:
            diff[0] = -1
        if diff[1] > 0:
            diff[1] = 1
        elif diff[1] < 0:
            diff[1] = -1
        next_node = robot_pos + diff
    new_robot_pos = next_node
    new_robot_pos_num = index_to_num(env_map,new_robot_pos)
    if not new_robot_pos_num in sparse_map:
        add_node(env_map,sparse_map,new_robot_pos_num,0)
        successA = add_edge(env_map,sparse_map,new_robot_pos_num,goal_node_num)
        if not successA:
            if debug:
                print("Uh oh, edge not added, this may lead to issues")
            successB = add_edge(env_map,sparse_map,new_robot_pos_num,robot_pos_num)
        #print("NODE ADD")
        #print(new_robot_pos_num,num_to_index(env_map,new_robot_pos_num))
        #print(goal_node_num,num_to_index(env_map,goal_node_num))
        #print("SPARSE PRE")
        #for key,val in sparse_map.items():
        #    if not val:
        #        print(num_to_index(env_map,key),val)
        #    else:
        #        for key2,val2 in val.items():
        #            print(num_to_index(env_map,key),num_to_index(env_map,key2),val2)
        #print("SPARSE POST")
        #for key,val in sparse_map.items():
        #    if not val:
        #        print(num_to_index(env_map,key),val)
        #    else:
        #        for key2,val2 in val.items():
        #            print(num_to_index(env_map,key),num_to_index(env_map,key2),val2)
    if debug:
        global current_weight_for_debug
        print("TARGET at %s, ROBOT MOVETO: %s, expected distance %d"%(target_pos,new_robot_pos.astype(np.int),current_weight_for_debug))
    return new_robot_pos
def display_map_with_nodes(env_map,robot_pos,target_pos,pause=True,delete=True):
    global sparse_map
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    if not delete:
        f = plt.gcf()
        ax= plt.gca()
    else:
        f, ax = plt.subplots()
    ax.imshow( env_map.T, interpolation="none", cmap='gray_r', origin='lower', \
                 extent=(-0.5, env_map.shape[0]-0.5, -0.5, env_map.shape[1]-0.5) )
    ax.axis([-0.5, env_map.shape[0]-0.5, -0.5, env_map.shape[1]-0.5])
    ax.set_xlabel('x')
    ax.set_ylabel('y')  
    hr = ax.plot(robot_pos[0], robot_pos[1], 'bs')
    ht = ax.plot(target_pos[0], target_pos[1], 'rs')
    for nodeA in sparse_map:
        nodeA_i = num_to_index(env_map,nodeA)
        ax.scatter([nodeA_i[0]],[nodeA_i[1]],c="red",marker=4)
        for nodeB in sparse_map[nodeA]:
            nodeB_i = num_to_index(env_map,nodeB)
            l = mlines.Line2D([nodeA_i[0],nodeB_i[0]], [nodeA_i[1],nodeB_i[1]])
            ax.add_line(l)
    f.canvas.flush_events()
    plt.show()
    if pause:
        print("Components connected as shown")
        raw_input("Press Enter to Continue")
    if delete:
        plt.close()
def RESET():
    global first_iter
    first_iter = True
    
def robotplanner(env_map, robot_pos, target_pos):
    '''
        This utilizes random sampling with Lazy Significant Edge detection to find optimal paths,
        including situations with narrow entries
    
    '''
    global first_iter
    global sparse_map
    if first_iter:
        print("BEGIN INITIALIZATION")
        initialize_planner(env_map,robot_pos,target_pos)
        first_iter = False
        print("INIT_ DONE")
        #if debug_prompt:
        #    display_map_with_nodes(env_map,robot_pos,target_pos)
        
    newrobot_pos = next_node_reccomendation(env_map,robot_pos,target_pos,sparse_map)
    if debug_prompt:
        display_map_with_nodes(env_map,robot_pos,target_pos)
    #print("NEW POSITION")
    #print(robot_pos)
    #print(target_pos)
    return newrobot_pos.astype(np.int)

