def PolicyIterationSolver(mdp):
	"""
	Uses Howard's Policy Iteration Algorithm
	to find the optimal policy
	Finds and puts the policy into mdp.PiStar
	Finds and puts the value into mdp.VStar
	"""

	# initialize the policy randomly
	mdp.randomInitPolicy()

	# define the required argmax function
	argmax = lambda array : max(zip(array, range(len(array))))[1]

	# iterate while not converged
	while True:

		# set VPi
		mdp.getValueFunction()

		# get current policy
		currPi = [0]*mdp.S
		currPi[-1] = mdp.PiStar[-1]

		# iterate over all states to find better policy
		for s in range(mdp.S - int(mdp.type == "epsiodic")):
			currPi[s] = argmax([sum([mdp.T[s][a][s_]*(mdp.R[s][a][s_] + mdp.gamma*mdp.VStar[s_])\
						for s_ in range(mdp.S)]) for a in range(mdp.A)])

		# check if converged
		if currPi == mdp.PiStar:
			break

		# update mdp.PiStar
		for s in range(mdp.S):
			mdp.PiStar[s] = currPi[s]
