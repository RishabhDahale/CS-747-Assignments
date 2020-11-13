import matplotlib.pyplot as plt
import os
import numpy as np
import glob
import csv


algos = ["sarsa", "expsarsa", "qlearning"]

if not os.path.exists("./plots"):
    os.makedirs("./plots")


def makePlot(algos, title):
	for algo, label in algos:
	    episodes = []
	    files = glob.glob(f"./results/{algo}-*.csv")
	    for fileName in files:
	        f = open(fileName, "r")
	        csv_reader = csv.reader(f, delimiter=",")
	        episode = []
	        next(csv_reader)
	        for line in csv_reader:
	            episode.append(float(line[1]))
	        f.close()
	        episodes.append(episode)
	    episodes = np.array(episodes)
	    x = np.array(range(1, 1+episodes.shape[1]))
	    avg = np.mean(episodes, axis=0)
	    std = np.std(episodes, axis=0)
	    plt.plot(x, avg, label=label)
	    plt.fill_between(x, y1=avg-std, y2=avg+std, alpha=0.2)
	plt.legend(loc='best')
	plt.title(title)
	plt.grid()
	plt.savefig(f"./plots/{title}.png")
	plt.clf()


# simple gridworld
algos = (("sarsa-4-0", "4 move Gridworld"),
		 ("sarsa-8-0", "8 move Gridworld"),
		 ("sarsa-9-0", "9 move Gridworld"),
		 ("sarsa-8-1", "8 move GridWorld with wind"))
makePlot(algos, "Sarsa(0) Comparison")

# baseline algos comparison
algos = (("sarsa-4-0", "Sarsa(0)"),
		 ("expsarsa-4-0", "Expected Sarsa"),
		 ("qlearning-4-0", "Q Learning"))
makePlot(algos, "4-Move Gridworld Baseline")

# Simple GridWorld Plotting
# for algo in algos:
#     episodes = []
#     files = glob.glob(f"./results/{algo}-4-0-*.csv")
#     for fileName in files:
#         f = open(fileName, "r")
#         csv_reader = csv.reader(f, delimiter=",")
#         episode = []
#         next(csv_reader)
#         for line in csv_reader:
#             episode.append(float(line[1]))
#         f.close()
#         episodes.append(episode)
#     episodes = np.array(episodes)
#     x = np.array(range(1, 1+episodes.shape[1]))
#     avg = np.mean(episodes, axis=0)
#     std = np.std(episodes, axis=0)
#     plt.plot(x, avg, label=algo)
#     plt.fill_between(x, y1=avg-std, y2=avg+std, alpha=0.2)
# plt.legend(loc='best')
# plt.title("Windy Gridworld")
# plt.grid()
# plt.savefig("./plots/simple_windy_gridworld.png")
# plt.clf()

# # Windy Gridworld Kings Move
# for algo in algos:
#     episodes = []
#     files = glob.glob(f"./results/{algo}-8-0-*.csv")
#     for fileName in files:
#         f = open(fileName, "r")
#         csv_reader = csv.reader(f, delimiter=",")
#         episode = []
#         next(csv_reader)
#         for line in csv_reader:
#             episode.append(float(line[1]))
#         f.close()
#         episodes.append(episode)
#     episodes = np.array(episodes)
#     x = np.array(range(1, 1+episodes.shape[1]))
#     avg = np.mean(episodes, axis=0)
#     std = np.std(episodes, axis=0)
#     plt.plot(x, avg, label=algo)
#     plt.fill_between(x, y1=avg-std, y2=avg+std, alpha=0.2)
# plt.legend(loc='best')
# plt.title("Windy Gridworld Kings Move (8 actions)")
# plt.grid()
# plt.savefig("./plots/windy_gridworld_kings_move_8.png")
# plt.clf()

# for algo in algos:
#     episodes = []
#     files = glob.glob(f"./results/{algo}-9-0-*.csv")
#     for fileName in files:
#         f = open(fileName, "r")
#         csv_reader = csv.reader(f, delimiter=",")
#         episode = []
#         next(csv_reader)
#         for line in csv_reader:
#             episode.append(float(line[1]))
#         f.close()
#         episodes.append(episode)
#     episodes = np.array(episodes)
#     x = np.array(range(1, 1+episodes.shape[1]))
#     avg = np.mean(episodes, axis=0)
#     std = np.std(episodes, axis=0)
#     plt.plot(x, avg, label=algo)
#     plt.fill_between(x, y1=avg-std, y2=avg+std, alpha=0.2)
# plt.legend(loc='best')
# plt.title("Windy Gridworld Kings Move (9 actions)")
# plt.grid()
# plt.savefig("./plots/windy_gridworld_kings_move_9.png")
# plt.clf()

# # Stochastic move
# for algo in algos:
#     episodes = []
#     files = glob.glob(f"./results/{algo}-8-1-*.csv")
#     for fileName in files:
#         f = open(fileName, "r")
#         csv_reader = csv.reader(f, delimiter=",")
#         episode = []
#         next(csv_reader)
#         for line in csv_reader:
#             episode.append(float(line[1]))
#         f.close()
#         episodes.append(episode)
#     episodes = np.array(episodes)
#     x = np.array(range(1, 1+episodes.shape[1]))
#     avg = np.mean(episodes, axis=0)
#     std = np.std(episodes, axis=0)
#     plt.plot(x, avg, label=algo)
#     plt.fill_between(x, y1=avg-std, y2=avg+std, alpha=0.2)
# plt.legend(loc='best')
# plt.title("Windy Gridworld Kings Move (8 actions) Stochastic Wind")
# plt.grid()
# plt.savefig("./plots/windy_gridworld_kings_move_8_stochastic_wind.png")
# plt.clf()

# # Sarsa comparison for normal, king and stochastic
# for algo in ['sarsa-4-0', 'sarsa-8-0', 'sarsa-8-1']:
#     files = glob.glob(f"./results/{algo}-*.csv")
#     episodes = []
#     for fileName in files:
#         f = open(fileName, "r")
#         csv_reader = csv.reader(f, delimiter=",")
#         episode = []
#         next(csv_reader)
#         for line in csv_reader:
#             episode.append(float(line[1]))
#         f.close()
#         episodes.append(episode)
#     episodes = np.array(episodes)
#     x = np.array(range(1, 1+episodes.shape[1]))
#     avg = np.mean(episodes, axis=0)
#     std = np.std(episodes, axis=0)
#     plt.plot(x, avg, label=f"{algo}")
#     plt.fill_between(x, y1=avg-std, y2=avg+std, alpha=0.2)
# plt.legend(loc='best')
# plt.title("Sarsa Comparison")
# plt.grid()
# plt.savefig("./plots/sarsa_comparison.png")
# plt.clf()

