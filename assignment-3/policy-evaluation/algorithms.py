import random

# Contains Algorithms to approximate Value Function from a Trajectory

def td_one(trj):
	# every-visit td(1)

	V = [[0., 0.] for _ in range(trj.S)]

	frac = 0.75 # the length of path we want to consider
	lenn = int(len(trj.Tx)*frac)

	# calculate reward of the remaining trajectory
	reward = 0.
	for tx in trj.Tx[lenn:][::-1]:
		reward = (reward * trj.gamma) + tx[1]

	V[trj.Tx[lenn][0]][0] = reward
	V[trj.Tx[lenn][0]][1] = 1.

	for tx in trj.Tx[:lenn][::-1]:
		reward = (reward * trj.gamma) + tx[1]
		V[tx[0]][0] = ((V[tx[0]][0] * V[tx[0]][1]) + reward)/(V[tx[0]][1] + 1)
		V[tx[0]][1] += 1

	trj.V = [V[i][0] for i in range(trj.S)]

	return

def td_zero(trj):

	# initial guess of value function
	trj.V = [0.]*trj.S

	# repeat multiple times
	for _ in range(1000):

		# init time
		time = 0.

		# iterate over all steps
		for tx in trj.Tx:

			time += 1. # increment the time

			# update the Value Function
			trj.V[tx[0]] = trj.V[tx[0]] + (1/time)*(tx[1] + trj.gamma*(trj.V[tx[-1]]) - trj.V[tx[0]])

	return


def td_lambda(trj, lamda):

	# initial guess of value function
	trj.V = [0.]*trj.S

	# repeat multiple times
	for _ in range(10):

		# init time
		time = 0.

		# init E values
		E = [0.]*trj.S

		# iterate over all steps
		for tx in trj.Tx:

			time += 1. # increment the time
			
			delta = tx[1] + trj.gamma*(trj.V[tx[-1]]) - trj.V[tx[0]]
			E[tx[0]] += 1

			for s_ in range(trj.S):
				trj.V[s_] += (1/time)*(delta*E[s_])
				E[s_] *= (trj.gamma * lamda)

	return