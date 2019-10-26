import sys
from algorithms import *

class Trajectory():
	"""
	A simple class do define a trajectory.
	Each Transition is of the format [s, r, s']
	"""

	def __init__(self, filename):

		self.filename = filename
		self.S = None # Number of States
		self.A = None # Number of Actions
		self.gamma = None # Discount Factor
		self.Tx = [] # List of Transitions
		self.V = None # Approximated Value Function

	def readFile(self):

		all_lines = open(self.filename, 'r').readlines()

		# read first three lines for MDP info
		self.S = int(all_lines[0])
		self.A = int(all_lines[1])
		self.gamma = float(all_lines[2])

		# read rest of the lines for Tx
		for i in range(3, len(all_lines)-1):

			cur = all_lines[i].split("\t")
			nxt = all_lines[i+1].split("\t")

			self.Tx.append([int(cur[0]), float(cur[-1]), int(nxt[0])])

	def printTrajectory(self):

		print(self.S)
		print(self.A)
		print(self.gamma)

		for tx in self.Tx:
			print(tx)

	def printAns(self):

		for s in range(self.S):
			print(self.V[s])

filename = sys.argv[1]
episode = Trajectory(filename)
episode.readFile()

# td_zero(episode)
# td_one(episode)
td_lambda(episode, 0.5)
episode.printAns()