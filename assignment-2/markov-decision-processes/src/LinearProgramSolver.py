from pulp import *

def LinearProgramSolver(mdp):
	"""
	Uses Linear Programming to find V*.
	Finds and puts the values into mdp.VStar
	Finds and puts the policy into mdp.PiStar
	"""

	# initialize the problem
	problem = LpProblem("MDP", LpMinimize)

	# create an array of decision variables. (V values)
	decisionVariables = [LpVariable("V_%d"%s) for s in range(mdp.S)]

	# remove the last state if it is as episodic task
	if mdp.type == "episodic":
		decisionVariables[-1] = 0.

	# add the objective function to the problem
	problem += sum(decisionVariables), "Sum of V values"

	# add all the constraints to the problem
	for s in range(mdp.S):
		for a in range(mdp.A):
			# add the constraint for state s and action a
			problem += sum([mdp.T[s][a][s_]*(mdp.R[s][a][s_] + mdp.gamma*decisionVariables[s_]) for s_ in range(mdp.S)]) \
					<= decisionVariables[s], "Constraint, state %d, action %d"%(s,a)

	# problem.writeLP("mdp.lp")

	# solve the formulated problem
	problem.solve()

	# set the values of the MDP
	mdp.VStar = [None]*mdp.S
	for s, var in enumerate(decisionVariables):
		mdp.VStar[s] = value(var)

	# save the optimal policy from VStar into PiStar
	mdp.getOptimalPolicy()