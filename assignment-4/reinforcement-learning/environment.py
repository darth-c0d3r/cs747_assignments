class WindyGridworld:
	"""
	The class encapsulating the grid along with its functions.
	"""

	def __init__(self, rows, columns, start, goal, wind_speeds):
		"""
		rows = number of rows in the gridworld
		column = number of columns in the grid world
		start = the start state of the agent (row, column)
		goal = the goal state of the agent (row, column)
		wind_speeds = the array of length = len(columns) of wind speeds
		"""

		# initialize the properties of the gridworld
		self.rows = rows
		self.columns = columns
		self.start = start
		self.goal = goal
		self.wind_speeds = wind_speeds

		# make sure the initialization is correct
		assert all([self.start[0] < self.rows, self.start[1] < self.columns]), "Incorrect Start State"
		assert all([self.goal[0] < self.rows, self.goal[1] < self.columns]), "Incorrect Goal State"
		assert len(self.wind_speeds) == self.columns, "Wind Speeds length Incorrect"

		# initialize the possible actions with their results
		self.actions = {"N": (1,0), "S": (-1,0), "E": (0,1), "W": (0,-1)}

	def take_step(self, state, action):
		"""
		action is one of the actions in self.actions
		update the current state and return the reward
		"""
			
		# get the overall effect of the action and get the new state
		effect = self.actions[action][0] + self.wind_speeds[state[1]], self.actions[action][1]
		new_state = [None, None]
		new_state[0] = max(0, min(state[0] + effect[0], self.rows-1))
		new_state[1] = max(0, min(state[1] + effect[1], self.columns-1))

		# return the corresponding next state and reward
		if new_state == list(self.goal):
			return new_state, 0
		else:
			return new_state, -1

def getDefaultGridworld():
	"""
	return the default dridworld as described in the example 6.5
	"""
	rows = 7
	columns = 10
	start = (3,0)
	goal = (3,7)
	wind_speeds = (0,0,0,1,1,1,2,2,1,0)

	return WindyGridworld(rows, columns, start, goal, wind_speeds)
