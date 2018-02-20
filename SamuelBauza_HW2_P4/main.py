import numpy as np
from glob import glob
from src.find_shortest_path import DJI_shortest_path,AS_shortest_path,\
    WAS_shortest_path


OUTPUT_PRECISION = 3
COSTS_OUTPUT    = "output_costs.txt"
ITER_OUTPUT     = "output_numiters.txt"

DATA_DIR = "./src"
inf = float("inf")
def main():
    weight_data = []
    coord_data  = []
    weight_paths = glob("./src/input_*")
    for path in weight_paths:
        weight_data.append(load_weights(path))
        
    coord_paths = glob("./src/coords_*")
    for path in coord_paths:
        coord_data.append(load_coords(path))
    
    num_tests = len(weight_paths)
    
    results = {"cost":[],"iter":[]}
        
    for i in range(num_tests):
        [DJI_cost,DJI_iter] = DJI_shortest_path(weight_data[i],coord_data[i])
        [AS_cost,AS_iter]   = AS_shortest_path(weight_data[i],coord_data[i])
        [WAS_cost,WAS_iter] = WAS_shortest_path(weight_data[i],coord_data[i],epsilon=[2,3,4,5])
        
        cur_cost = []
        cur_iter = []
        
        cur_cost.append(DJI_cost)
        cur_cost.append(AS_cost)
        for val in WAS_cost:
            cur_cost.append(val)
        
        cur_iter.append(DJI_iter)
        cur_iter.append(AS_iter)
        for val in WAS_iter:
            cur_iter.append(val)
            
        results["cost"].append(cur_cost)
        results["iter"].append(cur_iter)
        
    save_data(results)
    
def load_weights(infile):
    file = open(infile,"r")
    num_vert = int(file.readline().replace("\n",""))
    start_vert = int(file.readline().replace("\n",""))-1
    end_vert = int(file.readline().replace("\n",""))-1
    
    outmat = np.ones([num_vert,num_vert])*inf
    
    for line in file:
        vals = line.replace("\n","").split(" ")
        outmat[int(vals[0])-1,int(vals[1])-1] = float(vals[2])
    return [np.array(outmat),start_vert,end_vert]

def load_coords(infile):
    file = open(infile,"r")
    coord_list = []
    for line in file:
        cur_coords = [0,0]
        vals = line.replace("\n","").split(" ")
        cur_coords[0] = float(vals[0])
        cur_coords[1] = float(vals[1])
        coord_list.append(np.array(cur_coords))
    return np.array(coord_list)
def save_data(results):
    costs   = results["cost"]
    iter    =  results["iter"]
    
    print("RESULTS")
    print(costs)
    print(iter)
    
    num_tests   = len(costs)
    num_methods = len(costs[0])
    
    file = open(COSTS_OUTPUT,"w")
    cur_str = ""
    for i in range(num_tests):
        for j in range(num_methods):
            cur_str += str("%.4f "%(costs[i][j]))
        cur_str = cur_str[:-1]+"\n"
    file.write(cur_str)
    file.close()
    
    file = open(ITER_OUTPUT,"w")
    cur_str = ""
    for i in range(num_tests):
        for j in range(num_methods):
            cur_str += str("%d "%(iter[i][j]))
        cur_str = cur_str[:-1]+"\n"
    file.write(cur_str)
    file.close()

if __name__ == "__main__":
    print("Entering code")
    import psutil as PS
    cpu_count = PS.cpu_count()
    if cpu_count > 1:
        print("Setting processor affinity (Should improve performance)")
        p = PS.Process()
        affs = p.cpu_affinity()
        p.cpu_affinity([affs[-2],affs[-1]])      
    main()
