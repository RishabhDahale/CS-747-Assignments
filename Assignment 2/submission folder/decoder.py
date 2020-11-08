import argparse
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--grid", help="Path to grid file")
parser.add_argument("--value_policy", help="Path to value and policy file given by planner.py")
args = parser.parse_args()

gridPath = args.grid
policyPath = args.value_policy


PI = []
with open(policyPath, 'r') as f:
    for line in f:
        action = float(line.strip().split(" ")[1])
        PI.append(action)

grid = []
with open(gridPath, "r") as f:
    for line in f:
        values = line.strip().split(" ")
        for i in range(len(values)):
            values[i] = int(values[i])
        grid.append(values)

grid = np.array(grid)
stateIndex = np.zeros_like(grid) - 1
index = 0
start = -1
end = -1
for i in range(grid.shape[0]):
    for j in range(grid.shape[1]):
        if grid[i][j]!=1:
            stateIndex[i][j]=index
            index+=1
        if grid[i][j]==2:
            start = stateIndex[i][j]
        if grid[i][j]==3:
            end = stateIndex[i][j]

stateIndex = np.array(stateIndex)

path = ""
currState = start
while currState!=end:
    action = PI[currState]
    ns = None
    if action==0:
        l = np.where(stateIndex==currState)
        i = l[0][0]
        j = l[1][0]
        ns = stateIndex[i+1][j]
        path+=" S"
    elif action==1:
        l = np.where(stateIndex == currState)
        i = l[0][0]
        j = l[1][0]
        ns = stateIndex[i][j+1]
        path += " E"
    elif action==2:
        l = np.where(stateIndex == currState)
        i = l[0][0]
        j = l[1][0]
        ns = stateIndex[i-1][j]
        path += " N"
    elif action==3:
        l = np.where(stateIndex == currState)
        i = l[0][0]
        j = l[1][0]
        ns = stateIndex[i][j-1]
        path += " W"
    if ns==-1:
        ns = ns + 0
    currState = ns

print(path.strip())
