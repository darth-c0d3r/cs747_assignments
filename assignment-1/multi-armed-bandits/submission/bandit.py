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

def round_robin(bandit, horizon):
	reward = 0
	for idx in range(horizon):
		reward += pullArm(bandit, idx%len(bandit))
	return reward

def epsilon_greedy(bandit, horizon, epsilon):
	rewards = [0]*len(bandit)
	argmax = lambda array: max(zip(array, range(len(array))))[1]
	for _ in range(horizon):
		rand = random.random()
		if rand <= epsilon:
			# choose random
			idx = random.randint(0, len(bandit)-1)
			rewards[idx] += pullArm(bandit, idx)
		else:
			# choose optim
			idx = argmax(rewards)
			rewards[idx] += pullArm(bandit, idx)
	return sum(rewards)

def ucb(bandit, horizon):
	counts = [0]*len(bandit)
	reward = [0]*len(bandit)
	argmax = lambda array: max(zip(array, range(len(array))))[1]
	# sample each arm once
	for idx in range(min(len(bandit), horizon)):
		counts[idx] += 1
		reward[idx] += pullArm(bandit, idx)
	# sample according to the ucb rule
	for time in range(len(bandit), horizon):
		idx = argmax([((float(r)/float(c))+math.sqrt(2*math.log(time)/float(c))) for c,r in zip(counts, reward)])
		counts[idx] += 1
		reward[idx] += pullArm(bandit, idx)
	return sum(reward)

def kl_ucb(bandit, horizon):
	return 0

def thompson_sampling(bandit, horizon):
	success = [0]*len(bandit)
	failure = [0]*len(bandit)
	argmax = lambda array: max(zip(array, range(len(array))))[1]
	for _ in range(horizon):
		scores = [0]*len(bandit)
		idx = argmax([random.betavariate(s+1, f+1) for s,f in zip(success, failure)])
		reward = pullArm(bandit, idx)
		success[idx] += reward
		failure[idx] += 1-reward
	return  sum(success)

# process all input arguments
# can be better implemented but since the 
# sequence of args is guaranteed, this will do
file_in = sys.argv[2]
algorithm = sys.argv[4]
seed = int(sys.argv[6])
epsilon = float(sys.argv[8])
horizon = int(sys.argv[10])

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

max_reward = max(bandit)*horizon
regret = max_reward - reward

# for convenience
# print("bandit",bandit)
# print("horizon",horizon)
# print("max_reward",max_reward)
# print("reward",reward)
# print("regret",regret)

# instance, algorithm, random seed, epsilon, horizon, REG
print("%s, %s, %d, %f, %d, %f"%(file_in, algorithm, seed, epsilon, horizon, regret))