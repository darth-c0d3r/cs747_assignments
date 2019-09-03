import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Use output file to make the required plots

def get_data_for_time(data, time):
	reward = 0.0
	count = 0.0
	for line in data:
		if int(line[4]) == time:
			reward += float(line[-1])
			count += 1
	return reward/count

def get_data_for_algo_eps(data, algo, eps):
	new_data = []
	hsh = {}
	for line in data:
		if line[1].strip() == algo and str(float(line[3])) == str(float(eps)):
			new_data.append(line)
	for time in horizons:
		hsh[time] = get_data_for_time(new_data, time)
	return hsh


def get_data_for_file(data, file):
	new_data = []
	hsh = {}
	for line in data:
		if line[0] == file:
			new_data.append(line)
	for algo, eps in algo_eps:
		hsh[algo+"_"+str(eps)] = get_data_for_algo_eps(new_data, algo, eps)
	return hsh

def get_data(data):
	"""
	main function : returns data for all instances in a dictionary
	hsh[file][algo_eps][time]
	"""
	hsh = {}
	for file in files:
		hsh[file] = get_data_for_file(data, file)
	return hsh


files = [ "../instances/i-1.txt", "../instances/i-2.txt", "../instances/i-3.txt"]
algo_eps = [("round-robin",0), ("epsilon-greedy",0.002), ("epsilon-greedy",0.02,), 
("epsilon-greedy",0.2), ("ucb",0), ("kl-ucb",0), ("thompson-sampling",0)]
horizons = [50,200,800,3200,12800,51200,204800]
seeds = [i for i in range(50)]

lines = open("OUTPUT", "r").readlines()
for i in range(len(lines)):
	lines[i] = lines[i].split(",")
hsh = get_data(lines)

for file in files:
	# create a different plot for each file
	plt.xlabel("Horizon")
	plt.ylabel("Regret")
	for algo, eps in algo_eps:
		# create a separate line for each algo_eps
		Y = []
		for time in horizons:
			# create a X, Y style list
			Y.append(hsh[file][algo+"_"+str(eps)][time])
		print(Y)
		# plot the list here
		if algo == "epsilon-greedy":
			plt.plot(horizons,Y,label=algo+" "+str(eps))
		else:
			plt.plot(horizons,Y,label=algo)
		# plt.yscale('log')
		plt.xscale('log')
	plt.legend(loc='upper left')
	plt.show()
	print("\n")		
