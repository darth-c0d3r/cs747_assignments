import sys
from MDP import MDP
from LinearProgramSolver import *

# read in the arguments
filename = sys.argv[1]
algorithm = sys.argv[2]

mdp = MDP(filename)
mdp.build()

LinearProgramSolver(mdp)
mdp.getOptimalPolicy()
mdp.printAns()