import argparse
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--grid", help='Grid file path')
args = parser.parse_args()
gridPath = args.grid


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
end = []
for i in range(grid.shape[0]):
    for j in range(grid.shape[1]):
        if grid[i][j]!=1:
            stateIndex[i][j]=index
            index+=1
        if grid[i][j]==2:
            start = stateIndex[i][j]
        if grid[i][j]==3:
            end.append(stateIndex[i][j])

stateIndex = np.array(stateIndex)
numActions = 4          # N=0, E=1, S=2, W=3

endstr = [str(i) for i in end]

print(f"numStates {index}")
print("numActions 4")
print(f"start {start}")
print("end " + " ".join(endstr))

for i in range(grid.shape[0]):
    for j in range(grid.shape[1]):
        if grid[i][j]!=1:
            p = 1/(int(grid[i-1][j]!=1) + int(grid[i+1][j]!=1) + int(grid[i][j+1]!=1) + int(grid[i][j-1]!=1))
            if stateIndex[i][j] not in end:
                if (grid[i-1][j]!=1):
                    print(f"transition {stateIndex[i][j]} 2 {stateIndex[i-1][j]} {int(stateIndex[i-1][j]==end)*10000} {p}")
                if (grid[i][j+1]!=1):
                    print(f"transition {stateIndex[i][j]} 1 {stateIndex[i][j+1]} {int(stateIndex[i][j+1]==end)*10000} {p}")
                if (grid[i+1][j]!=1):
                    print(f"transition {stateIndex[i][j]} 0 {stateIndex[i+1][j]} {int(stateIndex[i+1][j]==end)*10000} {p}")
                if (grid[i][j-1]!=1):
                    print(f"transition {stateIndex[i][j]} 3 {stateIndex[i][j-1]} {int(stateIndex[i][j-1]==end)*10000} {p}")

print("mdptype episodic")
print("discount  0.9")
