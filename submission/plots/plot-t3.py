import statistics
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

file = '../outputDataT3.txt'
algos = ['epsilon-greedy']
# epsilons = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.1, 0.15]
epsilons = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035]
d1, d2, d3 = {}, {}, {}
for i in range(len(epsilons)):
    d1[epsilons[i]] = defaultdict(list)
    d2[epsilons[i]] = defaultdict(list)
    d3[epsilons[i]] = defaultdict(list)

regrets = {'../instances/i-1.txt': d1}
           # '../instances/i-2.txt': d2,
           # '../instances/i-3.txt': d3}
with open(file, 'r') as f:
    for line in f:
        values = line.split(',')
        instance = values[0]; algo = values[1][1:]
        seed = float(values[2]); epsilon = float(values[3])
        horizon = int(values[4]); regret = float(values[5][:-1])
        if epsilon in epsilons:
            regrets[instance][epsilon][horizon].append(regret)
    print(regrets)
    print(len(regrets['../instances/i-1.txt'][epsilons[0]][100]))
    for ins in regrets.keys():
        for ep in epsilons:
            for h in regrets[ins][ep].keys():
                regrets[ins][ep][h] = [statistics.mean(regrets[ins][ep][h]), statistics.stdev(regrets[ins][ep][h])]


def plot(instance, regrets):
    for i in range(len(epsilons)):
        means = [regrets[instance][epsilons[i]][k][0] for k in regrets[instance][epsilons[i]].keys()]
        var = [regrets[instance][epsilons[i]][k][1] for k in regrets[instance][epsilons[i]].keys()]
        means = np.array(means); var = np.array(var)
        plt.plot(list(regrets[instance][epsilons[i]].keys()),
                 means, label=f'{instance.split("/")[-1].split(".")[0]} - {epsilons[i]}')
        # plt.fill_between(list(regrets['../instances/i-1.txt'][algos[i]].keys()),
        #                  means + var, means - var, alpha=0.2)
    plt.legend(loc='best')
    plt.xlabel('Horizon')
    plt.ylabel('Regret')
    plt.xscale('log')
    plt.grid(True, which="both")
    plt.savefig(f'{file.split(".")[-2][-1]}-{instance.split("/")[-1].split(".")[0]}.png')
    plt.show()


plot('../instances/i-1.txt', regrets)
plot('../instances/i-2.txt', regrets)
plot('../instances/i-3.txt', regrets)

'''
for i in range(len(algos)):
    means = [regrets['../instances/i-2.txt'][algos[i]][k][0] for k in regrets['../instances/i-2.txt'][algos[i]].keys()]
    var = [regrets['../instances/i-2.txt'][algos[i]][k][1] for k in regrets['../instances/i-2.txt'][algos[i]].keys()]
    means = np.array(means); var = np.array(var)
    plt.plot(list(regrets['../instances/i-2.txt'][algos[i]].keys()),
             means, label=f'Instance 2 - {algos[i]}')
    # plt.fill_between(list(regrets['../instances/i-1.txt'][algos[i]].keys()),
    #                  means+var, means - var, alpha=0.2)
plt.legend(loc='best')
plt.xlabel('Horizon')
plt.ylabel('Regret')
plt.xscale('log')
plt.grid(True, which="both")
plt.show()

for i in range(len(algos)):
    means = [regrets['../instances/i-3.txt'][algos[i]][k][0] for k in regrets['../instances/i-3.txt'][algos[i]].keys()]
    var = [regrets['../instances/i-3.txt'][algos[i]][k][1] for k in regrets['../instances/i-3.txt'][algos[i]].keys()]
    means = np.array(means); var = np.array(var)
    plt.plot(list(regrets['../instances/i-3.txt'][algos[i]].keys()),
             means, label=f'Instance 3 - {algos[i]}')
    # plt.fill_between(list(regrets['../instances/i-1.txt'][algos[i]].keys()),
    #                  means+var, means - var, alpha=0.2)
plt.legend(loc='best')
plt.xlabel('Horizon')
plt.ylabel('Regret')
plt.xscale('log')
plt.grid(True, which="both")
plt.show()
'''