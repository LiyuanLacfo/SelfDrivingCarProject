import numpy as np
class Line(object):
	def __init__(self, x1, y1, x2, y2):
		self.x1 = float(x1)
		self.y1 = float(y1)
		self.x2 = float(x2)
		self.y2 = float(y2)
		self.slope = self.computeSlope()
		self.bias = self.computeBias()

	def computeSlope(self):
		return (self.y1-self.y2)/(self.x1 - self.x2 + np.finfo(float).eps)

	def computeBias(self):
		return self.y1 - self.slope * self.x1
    
    