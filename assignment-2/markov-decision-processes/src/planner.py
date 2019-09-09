import sys
from MDP import MDP
from LinearProgramSolver import *
from PolicyIterationSolver import *

# read in the arguments
filename = sys.argv[1].strip()
algorithm = sys.argv[2].strip()

# initialize the mdp struct
mdp = MDP(filename)
mdp.build()

# solve the mdp using apt algo
if algorithm == "lp":
	LinearProgramSolver(mdp)
elif algorithm == "hpi":
	PolicyIterationSolver(mdp)

# print the mdp ans
mdp.printAns()
