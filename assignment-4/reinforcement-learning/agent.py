from environment import *
from util import *
import numpy

class Agent:
	"""
	The class encapsulating the agent and reinforcement algorithms
	"""

	def __init__(self, env):
		"""
		env is the Gridworld Object
		"""

		# initialize the agent
		self.env = env

		# get the int -> action mapping
		self.actions = []
		for action in self.env.actions:
			self.actions.append(action)
		self.actions.sort()

		self.Q = None

	def Sarsa(self, alpha, epsilon, num_episodes):
		"""
		alpha is the step size
		epsilon is the parameter for epsilon greedy
		num_episodes is the number of episodes for which to run sarsa
		"""

		# initialize the values to be plotted and the time to be 0
		values = [0]
		time = 0

		# initialize Q
		self.Q = numpy.zeros((self.env.rows, self.env.columns, len(self.actions)))
		for idx in range(len(self.actions)):
			self.Q[self.env.goal[0], self.env.goal[1], idx] = 0

		for episode in range(num_episodes):

			# set the current state to initial state
			self.curr_state = list(self.env.start)

			# get the action according to epsilon greedy
			action = self.epsilonGreedy(self.curr_state, epsilon)

			# repeat while terminal state isn't reached
			while self.curr_state != list(self.env.goal):

				# get the new state, reward and new action
				state_, reward = self.env.take_step(self.curr_state, self.actions[action])
				action_ = self.epsilonGreedy(state_, epsilon)

				# update the Q values
				self.Q[self.curr_state[0], self.curr_state[1], action] += \
				alpha * (reward + self.Q[state_[0], state_[1], action_] - self.Q[self.curr_state[0], self.curr_state[1], action])

				# update the current state and action
				self.curr_state = state_
				action = action_

				time += 1

			values.append(time)

		return values

	def epsilonGreedy(self, state, epsilon):
		"""
		Return the action to be taken at state using self.Q
		"""

		# define the required argmax function
		argmax = lambda array : max(zip(array, range(len(array))))[1]

		# sample a uniform random number between 0 and 1
		rnd = numpy.random.uniform(0,1)

		# return the required action
		if rnd < epsilon:
			return numpy.random.choice(len(self.actions), 1)[0]
		else:
			return argmax(list(self.Q[state[0], state[1], :]))

def main():

	env = getDefaultGridworld()
	agent = Agent(env)
	alpha = 0.5
	epsilon = 0.1
	num_episodes = 170
	values = agent.Sarsa(alpha, epsilon, num_episodes)

	makePlot(values)

main()