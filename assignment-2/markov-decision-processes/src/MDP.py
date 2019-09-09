import sys
import random
from pulp import *

class MDP:
	"""
	Wrapper Class for representing MDPs.
	self.S = number of states
	self.A = number of actions
	self.T = transition function
	self.R = reward function
	self.gamma = discount factor
	self.type = continuing or episodic

	if self.type = episodic, then state S-1
	will be considered a terminal state
	"""
	
	def __init__(self, filename):

		super(MDP, self).__init__()
		self.filename = filename

		# representative variables
		self.S = None
		self.A = None
		self.T = None
		self.R = None
		self.gamma = None
		self.type = None

		# calculated variables
		# Note that at intermittent stages,
		# VStar amd PiStar need not be optimum
		self.VStar = None # optimal values
		self.PiStar = None # optimal policy

	def build(self):
		"""
		reads self.filename and generates a dict.
		"""
		# get all the lines in the file
		all_lines = open(self.filename).readlines()

		# get number of states and number of actions
		self.S = int(all_lines[0].strip())
		self.A = int(all_lines[1].strip())

		line_idx = 2
		# get the reward function
		self.R = [ [ [None for _ in range(self.S)] for _ in range(self.A)] for _ in range(self.S)]
		for s in range(self.S):
			for a in range(self.A):
				curr_line = all_lines[line_idx].strip().split("\t")
				curr_line = [float(val) for val in curr_line]
				line_idx += 1
				for s_ in range(self.S):
					self.R[s][a][s_] = curr_line[s_]

		# get the transition function
		self.T = [ [ [None for _ in range(self.S)] for _ in range(self.A)] for _ in range(self.S)]
		for s in range(self.S):
			for a in range(self.A):
				curr_line = all_lines[line_idx].strip().split("\t")
				curr_line = [float(val) for val in curr_line]
				line_idx += 1
				for s_ in range(self.S):
					self.T[s][a][s_] = curr_line[s_]

		# get gamma and the type of the MDP
		self.gamma = float(all_lines[-2].strip())
		self.type = all_lines[-1].strip()

	def print(self, filename=None):
		"""
		prints the MDP in exact same format
		if filename is not None, output will
		be written into that file
		"""

		# redirect output to filename if required
		og_stdout = sys.stdout
		if filename is not None:
			sys.stdout = open(filename, 'w+')

		print(self.S)
		print(self.A)

		for s in range(self.S):
			for a in range(self.A):
				for s_ in range(self.S):
					print(str(self.R[s][a][s_]), end="\t")
				print()

		for s in range(self.S):
			for a in range(self.A):
				for s_ in range(self.S):
					print(str(self.T[s][a][s_]), end="\t")
				print()

		print(self.gamma)
		print(self.type)

		# set the output redirection to default
		sys.stdout = og_stdout

	def randomInitPolicy(self):
		"""
		initializes PiStar as a random Policy
		"""
		self.PiStar = [random.randint(0,self.A-1) for _ in range(self.S)]


	def getOptimalPolicy(self):
		"""
		calculates the optimal policy given
		that VStar is already calculated
		"""

		# initialize the PiStar array
		self.PiStar = [None]*self.S

		# define the argmax and the QValue function
		argmax = lambda array : max(zip(array, range(len(array))))[1]
		QValue = lambda s, a, s_ : self.T[s][a][s_]*(self.R[s][a][s_] + self.gamma*self.VStar[s_])

		# iterate over all the states
		for s in range(self.S):
			QValues = [sum([QValue(s,a,s_) for s_ in range(self.S)]) for a in range(self.A)]
			self.PiStar[s] = argmax(QValues)

	def getValueFunction(self):
		"""
		calculates the value function given
		that PiStar is already calculated
		Note : PiStar need not be optimal
		It will calculate VPi and set VStar = VPi
		# use Pulp library to get VStar
		"""

		# initialize VStar if not already done
		if self.VStar is None:
			self.VStar = [None]*self.S

		# declare the problem variable
		problem = LpProblem()

		# make an array of S decision variables
		decisionVariables = [LpVariable("V_%d"%s) for s in range(self.S)]

		# remove the last state if it is as episodic task
		if self.type == "episodic":
			decisionVariables[-1] = 0.


		# add the set of S equations to the problem
		for s in range(self.S):
			a = self.PiStar[s]
			problem += sum([self.T[s][a][s_]*(self.R[s][a][s_] + self.gamma*decisionVariables[s_]) for s_ in range(self.S)]) \
					== decisionVariables[s], "Bellman's Equation, State %d"%s

		# solve the problem
		problem.solve()

		# set the values in self.VStar
		for s, var in enumerate(decisionVariables):
			self.VStar[s] = value(var)

	def printAns(self, filename=None):
		"""
		prints the final answer into the file or to stdout if file is None
		"""

		# redirect output to filename if required
		og_stdout = sys.stdout
		if filename is not None:
			sys.stdout = open(filename, 'w+')

		# iterate over all states
		for s in range(self.S):
			print("%06f\t%d"%(self.VStar[s], self.PiStar[s]))

		# set the output redirection to default
		sys.stdout = og_stdout


	