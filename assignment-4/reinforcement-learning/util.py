from matplotlib import pyplot

def makePlot(values, title=""):
	"""
	plot timestep values with respect to episodes
	"""

	pyplot.plot(values, range(len(values)))
	pyplot.xlabel("Time Steps")
	pyplot.ylabel("Episodes")
	pyplot.title(title)
	pyplot.show()