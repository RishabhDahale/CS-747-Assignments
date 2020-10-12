import argparse
import numpy as np
import pulp
from copy import deepcopy
from collections import defaultdict


class MDP:
    def __init__(self, mdpPath):
        """
        Class to define MDP
        MDP is defined with (S, A, T, R, gamma). Reading these paramteres from the file  mdpPath
        :param mdpPath:
        """
        self.path = mdpPath
        self.S, self.A = None, None
        self.R, self.T = None, None  # It will be a 3d matrix with dimensions as [currState][action][nextState]

        with open(self.path, 'r') as f:
            for line in f:
                values = line.split(" ")
                if values[0] == 'numStates':
                    self.S = int(values[1][:-1])
                elif values[0] == 'numActions':
                    self.A = int(values[1][:-1])
                elif values[0] == 'start':
                    self.startState = int(values[1][:-1])
                elif values[0] == 'end':
                    for i in range(1, len(values)):
                        if i != (len(values) - 1):
                            values[i] = int(values[i])
                        else:
                            values[i] = int(values[i][:-1])
                    self.endStates = deepcopy(values[1:])
                elif values[0] == 'transition':
                    cs = int(values[1]);
                    a = int(values[2]);
                    ns = int(values[3])
                    r = float(values[4]);
                    p = float(values[5][:-1])
                    self.R[cs][a][ns] = r
                    self.T[cs][a][ns] = p
                elif values[0] == 'mdptype':
                    self.type = values[1][:-1]
                elif values[0] == 'discount':
                    self.gamma = float(values[-1][:-1].strip())

                if (self.S is not None) and (self.A is not None) and (self.R is None) and (self.T is None):
                    self.R, self.T = np.zeros((self.S, self.A, self.S)), np.zeros((self.S, self.A, self.S))

        # print(f"MDP, #states={self.S}, #actions={self.A}, gamma={self.gamma}, type={self.type}")
        # print("\nTransitions rewards:", self.R)
        # print("\nTransition Probabilities:", self.T)


def ValueIteration(mdp: MDP, errorLimit=1e-6):
    stateValue = np.zeros((mdp.S, 1))

    def iteration(stateValue):
        updatedStateValue = np.zeros_like(stateValue)
        PI = np.zeros((mdp.S, 1)) - 1
        for s in range(mdp.S):
            t = np.sum(mdp.T[s] * (mdp.R[s] + (mdp.gamma * np.transpose(stateValue))), axis=1)
            updatedStateValue[s] = np.max(t)
            PI[s] = np.argmax(t)
        return updatedStateValue, PI

    newStateValue, pi = iteration(stateValue)
    while np.linalg.norm(newStateValue - stateValue, 1) >= (errorLimit * mdp.S):
        stateValue = deepcopy(newStateValue)
        newStateValue, pi = iteration(stateValue)

    for i in range(newStateValue.shape[0]):
        print(f"{newStateValue[i][0]} {pi[i][0]}")
    return newStateValue, pi


def lpSolution(mdp: MDP):
    lpProb = pulp.LpProblem("MDP_solution", pulp.LpMinimize)
    stateValues = pulp.LpVariable.dicts(name='V', indexs=list(range(mdp.S)), cat=pulp.LpContinuous)

    # Objective function
    lpProb += pulp.lpSum(stateValues)

    # Constraints
    for s in range(mdp.S):
        for a in range(mdp.A):
            lpProb += pulp.lpSum(
                [mdp.T[s][a][i] * (mdp.R[s][a][i] + (mdp.gamma * stateValues[i])) for i in range(mdp.S)]) <= \
                      stateValues[s]

    status = lpProb.solve(solver=pulp.PULP_CBC_CMD(msg=False, threads=1))
    assert pulp.LpStatus[status] == 'Optimal'
    V, pi = np.zeros((mdp.S, 1)), np.zeros((mdp.S, 1))
    for s in range(mdp.S):
        V[s] = pulp.value(stateValues[s])
    pi = np.zeros_like(V)
    for s in range(mdp.S):
        pi[s] = np.argmax(np.sum(mdp.T[s] * (mdp.R[s] + (mdp.gamma * np.transpose(V))), axis=1))
        print(f"{V[s][0]} {pi[s][0]}")
    return V, pi


def hpi(mdp: MDP):
    # At each time step calculate V^{pi}(s)
    PI = np.random.choice(mdp.S, size=(mdp.S))

    def getVpi(PI):
        # For the given policy pi, to get the value function values, we need to solve the system of linear equations
        # given by bellman equations. Let the syatem of linear equations be Ax=y. then x = A^{-1}y
        A = np.zeros((mdp.S, mdp.S))
        y = np.zeros((mdp.S, 1))
        for s in range(mdp.S):
            t = np.zeros(mdp.S)
            r = np.zeros(mdp.S)
            for i in range(mdp.S):
                if i != s:
                    A[s][i] = -1 * mdp.T[s][PI[s]][i] * mdp.gamma
                else:
                    A[s][i] = 1 - (mdp.T[s][PI[s]][i] * mdp.gamma)
                r[i] = mdp.R[s][PI[s]][i]
                t[i] = mdp.T[s][PI[s]][i]
            y[s][0] = np.sum(r * t)
        x = np.linalg.inv(A) @ y
        return x.reshape(x.shape[0])

    def getQvals(v):
        q = np.zeros((mdp.S, mdp.A))
        for s in range(mdp.S):
            for a in range(mdp.A):
                q[s][a] = np.sum(mdp.T[s][a] * (mdp.R[s][a] + (mdp.gamma * v)))
        return q

    def getIA(Q, V):
        IA = defaultdict(list)
        IS = []
        for s in range(mdp.S):
            for a in range(mdp.A):
                if Q[s][a] > V[s]:
                    IA[s].append(a)
            if len(IA[s])>=1:
                IS.append(s)
        return IA, IS

    V = getVpi(PI)
    Q = getQvals(V)
    IA, IS = getIA(Q, V)

    while len(IS) != 0:
        for s in IS:
            PI[s] = np.random.choice(IA[s])
        V = getVpi(PI)
        Q = getQvals(V)
        IA, IS = getIA(Q, V)

    for i in range(mdp.S):
        print(f"{V[i]} {PI[i]}")

# mdp = MDP("simpleMdpE.txt")
# print(f"#States={mdp.S}, #actions={mdp.A}")
# hpi(mdp)

parser = argparse.ArgumentParser()
parser.add_argument("--mdp", help="Path to input mdp file. Use complete path. No relative path")
parser.add_argument("--algorithm", choices=['vi', 'hpi', 'lp'], help='Algorithm to be used for solving')

args = parser.parse_args()
algo = args.algorithm

mdp = MDP(args.mdp)
if algo=='vi':
    ValueIteration(mdp, errorLimit=1e-8)
elif algo=='hpi':
    hpi(mdp)
elif algo=='lp':
    lpSolution(mdp)


