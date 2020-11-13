To get the results of the experiments for 20 random seeds run "runner.sh" script. It will create a directory "results" where all the csv files would be stored. runner.sh will also call plot.py script which will read the data from these csv files and make appropriate plots.

main.py
	main.py is the one and only script needed to run all the experiments. It takes some parameters from the command line
	--seed SEED           			random seed of the experiments
	--moves MOVES         			Maximum number of moves possible from a position. For
	                      			normal windy gridworld this should be 4, for kings
	                      			will be 8 or 9  (default value 4)
	--windDev WINDDEV     			Deviation of the wind. Should be integer. For stochastic
						case, give 1 as input (default value 0)
	--maxTimeSteps MAXTIMESTEPS 		Maximum Time Steps to allow the run 	(default 8000)
	--epsilon EPSILON     			Probability with which exploration should happen (default 0.1)
	--alpha ALPHA         			Learning rate			(default 0.5)
	--algo {sarsa,expsarsa,qlearning}	Algorithm to be used for 	(default sarsa)
	--gamma GAMMA         			Discount factor to be used 	(default 1)
	--endReward ENDREWARD 			Reward for the end state	(default 1)

plot.py
	Reads data from the "results" folder, generated appropriate plots and stores them in the folder "plots"
