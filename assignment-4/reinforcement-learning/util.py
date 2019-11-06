from matplotlib import pyplot

def makePlot(values, title=-1):
	"""
	plot timestep values with respect to episodes
	"""

	titles = ["Task 1: Deterministic Model without King's moves", "Task 2: Deterministic with King's moves",\
			 "Task 3: Stochastic Model with King's moves",""]

	pyplot.plot(values, range(len(values)))
	pyplot.xlabel("Time Steps")
	pyplot.ylabel("Episodes")
	pyplot.title(titles[title])
	pyplot.show()