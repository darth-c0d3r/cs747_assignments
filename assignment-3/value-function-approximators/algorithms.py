# Contains Algorithms to approximate Value Function from a Trajectory

def td_zero(trj):

	# initial guess of value function
	trj.V = [0.]*trj.S

	# init time
	time = 0.

	# iterate over all steps
	for tx in trj.Tx:

		time += 1. # increment the time

		# update the Value Function
		trj.V[tx[0]] = trj.V[tx[0]] + (1/time)*(tx[1] + trj.lamda*(trj.V[tx[-1]]) - trj.V[tx[0]])

	return