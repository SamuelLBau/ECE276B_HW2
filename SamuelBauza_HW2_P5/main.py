import numpy as np
import math
from numpy import loadtxt
import matplotlib.pyplot as plt
plt.ion()
import time
import sys

from robotplanner import robotplanner,RESET
from targetplanner import targetplanner

def tic():
  return time.time()
def toc(tstart, nm=""):
  print('%s took: %s sec.\n' % (nm,(time.time() - tstart)))


def runtest(mapfile, robotstart, targetstart):
  # current positions of the target and robot
  robotpos = np.copy(robotstart);
  targetpos = np.copy(targetstart);
  
  # environment
  envmap = loadtxt(mapfile)
    
  # draw the environment
  # transpose because imshow places the first dimension on the y-axis
  f, ax = plt.subplots()
  ax.imshow( envmap.T, interpolation="none", cmap='gray_r', origin='lower', \
             extent=(-0.5, envmap.shape[0]-0.5, -0.5, envmap.shape[1]-0.5) )
  ax.axis([-0.5, envmap.shape[0]-0.5, -0.5, envmap.shape[1]-0.5])
  ax.set_xlabel('x')
  ax.set_ylabel('y')  
  hr = ax.plot(robotpos[0], robotpos[1], 'bs')
  ht = ax.plot(targetpos[0], targetpos[1], 'rs')
  f.canvas.flush_events()
  plt.show()
  
  follow_icon = False

  # now comes the main loop
  numofmoves = 0
  caught = False
  for i in range(99999):
    # call robot planner
    t0 = tic()
    newrobotpos = robotplanner(envmap, robotpos, targetpos)
    # compute move time for the target
    act_time = (tic()-t0)/2.0
    movetime = max(1, int(math.ceil(act_time)))
    sys.stdout.write('move_num = %d, move time: %d, actual time %f\r'%(i,movetime,act_time))
    sys.stdout.flush()
    
    #check that the new commanded position is valid
    if ( newrobotpos[0] < 0 or newrobotpos[0] >= envmap.shape[0] or \
         newrobotpos[1] < 0 or newrobotpos[1] >= envmap.shape[1] ):
      raise Exception('ERROR: out-of-map robot position commanded\n')
      break
    elif ( envmap[newrobotpos[0], newrobotpos[1]] != 0 ):
      raise Exception('ERROR: invalid robot position commanded\n')
      break
    elif (abs(newrobotpos[0]-robotpos[0]) > 1 or abs(newrobotpos[1]-robotpos[1]) > 1):
      raise Exception('ERROR: invalid robot move commanded\n')
      break

    # call target planner to see how they move within the robot planning time
    newtargetpos = np.array([5,4998])#targetplanner(envmap, robotpos, targetpos, targetstart, movetime)
    
    # make the moves
    robotpos = newrobotpos
    targetpos = newtargetpos
    numofmoves += 1
    
    # draw positions
    if(follow_icon):
        hr[0].set_xdata(robotpos[0])
        hr[0].set_ydata(robotpos[1])
        ht[0].set_xdata(targetpos[0])
        ht[0].set_ydata(targetpos[1])
        f.canvas.flush_events()
        plt.show()
        
    # check if target is caught
    if (abs(robotpos[0]-targetpos[0]) <= 1 and abs(robotpos[1]-targetpos[1]) <= 1):
      print('robotpos = (%d,%d), targetpos = (%d,%d)' %(robotpos[0],robotpos[1],targetpos[0],targetpos[1]))
      caught = True
      break

  return caught, numofmoves


def test_map0():
  robotstart = np.array([0, 2])
  targetstart = np.array([5, 3])
  return runtest('src/map0.txt', robotstart, targetstart)

def test_map1():
  robotstart = np.array([699, 799])
  targetstart = np.array([699, 1699])
  return runtest('src/map1.txt', robotstart, targetstart)

def test_map2():
  robotstart = np.array([0, 2])
  targetstart = np.array([7, 9])
  return runtest('src/map2.txt', robotstart, targetstart)
  
def test_map3():
  robotstart = np.array([249, 249])
  targetstart = np.array([399, 399])
  return runtest('src/map3.txt', robotstart, targetstart)

def test_map4():
  robotstart = np.array([0, 0])
  targetstart = np.array([5, 6])
  return runtest('src/map4.txt', robotstart, targetstart)

def test_map5():
  robotstart = np.array([0, 0])
  targetstart = np.array([29, 59])
  return runtest('src/map5.txt', robotstart, targetstart)

def test_map6():
  robotstart = np.array([0, 0])
  targetstart = np.array([29, 36])
  return runtest('src/map6.txt', robotstart, targetstart)

def test_map7():
  robotstart = np.array([1, 1])
  targetstart = np.array([4998, 4998])
  return runtest('src/map7.txt', robotstart, targetstart)


def test_map1b():
  robotstart = np.array([249, 1199])
  targetstart = np.array([1649, 1899])
  return runtest('src/map1.txt', robotstart, targetstart)
  
def test_map1c():
  robotstart = np.array([249, 1199])
  targetstart = np.array([1000, 1250])
  return runtest('src/map1.txt', robotstart, targetstart)

def test_map3b():
  robotstart = np.array([74, 249])
  targetstart = np.array([399, 399])
  return runtest('src/map3.txt', robotstart, targetstart)

def test_map3c():
  robotstart = np.array([4, 399])
  targetstart = np.array([399, 399])
  return runtest('src/map3.txt', robotstart, targetstart)
  

if __name__ == "__main__":
    #test_list       = [test_map0,test_map1  ,test_map1b ,test_map1c ,test_map2,test_map3,test_map3b ,test_map3c ,test_map4,test_map5,test_map6,test_map7]
    #expected_result = ["5"      ,"1200"     ,"2700"     ,"FAIL"     ,"Unknown","Unknown","Unknown"  ,"Unknown"  ,"Unknown","Unknown","Unknown","Unknown"]
    #test_str       =  ["test_map0","test_map1"  ,"test_map1b" ,"test_map1c" ,"test_map2","test_map3","test_map3b" ,"test_map3c" ,"test_map4","test_map5","test_map6","test_map7"]
    test_list = [test_map7]
    expected_result = ["13000"]
    test_str = ["test_map7"]
    for id,test in enumerate(test_list):
        print("\n------------------------------\nTest(%d) %s Expected result V (Approximate) %s\n--------------------------------"%(id,test_str[id],expected_result[id]))
        try:
            print("Beginning test %d"%(id))
            RESET()
            caught, numofmoves = test()
            print('TEST {}, Number of moves made: {}; Target caught: {}.'.format(id,numofmoves, caught))
        except Exception as e:
            print("Test %d error, cause %s"%(id,e.args[0]))
        

