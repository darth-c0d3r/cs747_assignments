import sys
import random
import math

# https://lemire.me/blog/2004/11/25/computing-argmax-fast-in-python/

def readBandit(filename):
	"""
	read the file containing the bandit instance and return a bandit instance (list).
	"""
	file = open(filename, 'r')
	lines = file.readlines()
	bandit = [float(value.strip()) for value in lines]
	return bandit

def pullArm(bandit, arm):
	"""
	pull the arm of the bandit and return the result.
	"""
	if arm >= len(bandit):
		print("Arm exceeds the limit of Bandit.")
		return -1
	p = bandit[arm]
	probe = random.random()
	if probe <= p:
		return 1
	return 0

def binary_search(c,p,l,r,eps=1e-4):
	"""
	binary search to find q value for kl-ucb
	"""
	q = (l+r)/2.0
	if r-l <= eps:
		return q

	val = (p)*math.log(q) + math.log(1-q) - (p)*math.log(1-q) - c

	# for a monotonously decreasing function
	if val > 0:
		return binary_search(c,p,q,r,eps)
	else:
		return binary_search(c,p,l,q,eps)

def round_robin(bandit, horizon):

	curr_idx = 0
	ans = []
	reward = 0

	for idx in range(horizon[-1]):

		reward += pullArm(bandit, idx%len(bandit)) # add to current reward

		#	 if reward for time t is to be returned
		if idx+1 == horizon[curr_idx]:
			curr_idx += 1
			ans.append(reward)

	return ans

def epsilon_greedy(bandit, horizon, epsilon):

	curr_idx = 0
	ans = []
	reward = [0]*len(bandit)
	counts = [0]*len(bandit)
	argmax = lambda array: max(zip(array, range(len(array))))[1]

	for t in range(horizon[-1]):

		# random probe
		rand = random.random()

		if rand <= epsilon:
			# choose random
			idx = random.randint(0, len(bandit)-1)
			reward[idx] += pullArm(bandit, idx) # add to reward of idx
			counts[idx] += 1
		else:
			# choose optim
			idx = argmax([0.5 if c == 0 else float(r)/float(c) for r,c in zip(reward,counts)])
			reward[idx] += pullArm(bandit, idx) # add to reward of idx
			counts[idx] += 1

		# if reward for time t is to be returned
		if t+1 == horizon[curr_idx]:
			curr_idx += 1
			ans.append(sum(reward))

	return ans

def ucb(bandit, horizon):

	curr_idx = 0
	ans = []
	counts = [0]*len(bandit)
	reward = [0]*len(bandit)
	argmax = lambda array: max(zip(array, range(len(array))))[1]

	# sample each arm once
	for idx in range(min(len(bandit), horizon[-1])):
		counts[idx] += 1
		reward[idx] += pullArm(bandit, idx) # add to reward of idx
		if idx+1 == horizon[curr_idx]:
			curr_idx += 1
			ans.append(sum(reward))

	# sample according to the ucb rule
	for time in range(len(bandit), horizon[-1]):

		# argmax accoriding to the UCB rule
		idx = argmax([((float(r)/float(c))+math.sqrt(2*math.log(time)/float(c))) for c,r in zip(counts, reward)])
		counts[idx] += 1
		reward[idx] += pullArm(bandit, idx) # add to reward of idx

		# if reward for time t is to be returned
		if time+1 == horizon[curr_idx]:
			curr_idx += 1
			ans.append(sum(reward))

	return ans

def kl_ucb(bandit, horizon):

	curr_idx = 0
	ans = []
	counts = [0]*len(bandit)
	reward = [0]*len(bandit)
	argmax = lambda array: max(zip(array, range(len(array))))[1]

	# sample each arm once
	for idx in range(min(len(bandit), horizon[-1])):
		counts[idx] += 1
		reward[idx] += pullArm(bandit, idx) # add to reward of idx
		if idx+1 == horizon[curr_idx]:
			curr_idx += 1
			ans.append(sum(reward))

	# sample according to the kl-ucb rule
	for time in range(len(bandit), horizon[-1]):
		ucb_val = [0]*len(bandit)

		for idx in range(len(bandit)):
			# calculate extra values needed
			p = float(reward[idx])/float(counts[idx])
			k = 0.0
			if reward[idx] != 0 and reward[idx] != counts[idx]:
				k = (p)*math.log(p) + (1-p)*math.log(1-p)
			c = k-((math.log(time) + 3*math.log(math.log(time)))/float(counts[idx]))
			ucb_val[idx] = binary_search(c,p,p,1)

		idx = argmax(ucb_val)
		counts[idx] += 1
		reward[idx] += pullArm(bandit, idx) # add to reward of idx

		# if reward for time t is to be returned
		if time+1 == horizon[curr_idx]:
			curr_idx += 1
			ans.append(sum(reward))

	return ans

def thompson_sampling(bandit, horizon):

	curr_idx = 0
	ans = []
	success = [0]*len(bandit)
	failure = [0]*len(bandit)
	argmax = lambda array: max(zip(array, range(len(array))))[1]

	for t in range(horizon[-1]):
		scores = [0]*len(bandit)
		idx = argmax([random.betavariate(s+1, f+1) for s,f in zip(success, failure)])
		reward = pullArm(bandit, idx) # add to reward of idx
		success[idx] += reward
		failure[idx] += 1-reward

		# if reward for time t is to be returned
		if t+1 == horizon[curr_idx]:
			curr_idx += 1
			ans.append(sum(success))

	return ans

def get_results(file_in, algorithm, seed, epsilon, horizon):

	# set the seed
	random.seed(seed)

	# read the bandit file
	bandit = readBandit(file_in)

	# call the corresponding algorithm
	reward = None
	# round-robin, epsilon-greedy, ucb, kl-ucb, and thompson-sampling
	if algorithm == "round-robin":
		reward = round_robin(bandit, horizon)
	elif algorithm == "epsilon-greedy":
		reward = epsilon_greedy(bandit, horizon, epsilon)
	elif algorithm == "ucb":
		reward = ucb(bandit, horizon)
	elif algorithm == "kl-ucb":
		reward = kl_ucb(bandit, horizon)
	elif algorithm == "thompson-sampling":
		reward = thompson_sampling(bandit, horizon)

	# calculate regret
	regret = [max(bandit)*h - r for h,r in zip(horizon,reward)]

	# for convenience
	# print("bandit",bandit)
	# print("horizon",horizon)
	# print("max_reward",max_reward)
	# print("reward",reward)
	# print("regret",regret)

	# instance, algorithm, random seed, epsilon, horizon, REG
	for h,r in zip(horizon, regret):
		print("%s, %s, %d, %f, %d, %f"%(file_in, algorithm, seed, epsilon, h, r))

if __name__ == "__main__":
	# process all input arguments
	# can be better implemented but since the 
	# sequence of args is guaranteed, this will do
	file_in = sys.argv[2]
	algorithm = sys.argv[4]
	seed = int(sys.argv[6])
	epsilon = float(sys.argv[8])
	horizon = [int(sys.argv[10])]
	get_results(file_in, algorithm, seed, epsilon, horizon)
