import statistics
from collections import defaultdict


with open("outputDataT2-v2.txt", 'r') as f:
	# means = defaultdict(list)
	means = {"../instances/i-1.txt": defaultdict(list),
			 "../instances/i-2.txt": defaultdict(list),
			 "../instances/i-3.txt": defaultdict(list)}
	for line in f:
		values = line.split(",")
		ins = values[0]
		reg = float(values[-1][:-1])
		horizon = float(values[-2])
		means[ins][horizon].append(reg)
	for k in means.keys():
		for h in means[k].keys():
			means[k][h] = statistics.mean(means[k][h])
	print(means)

