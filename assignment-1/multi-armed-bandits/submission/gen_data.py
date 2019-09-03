# just some utility code that calls the get_results function
# in the bandit file to get the required values

from bandit import get_results

files = [ "../instances/i-1.txt", "../instances/i-2.txt", "../instances/i-3.txt"]
algo_eps = [("round-robin",0), ("epsilon-greedy",0.002), ("epsilon-greedy",0.02,), 
("epsilon-greedy",0.2), ("ucb",0), ("kl-ucb",0), ("thompson-sampling",0)]
horizons = [50,200,800,3200,12800,51200,204800]
seeds = [i for i in range(50)]

for seed in seeds:
	for algo, eps in algo_eps:
		for file in files:
			get_results(file, algo, seed, eps, horizons)