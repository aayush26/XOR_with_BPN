import numpy as np

class BackPropagationNetwork:
	layerCount = 0
	shape = None												# tuple
	weights = []

	def __init__(self, layerSize):								# layerSize -> tuple
		#Initialise network

		#Layer info
		self.layerCount = len(layerSize)-1						# - 1 because no. of layers will be 1 less than the layerSize
		self.shape = layerSize

		# Input/Output data from the last run
		self._layerInput = []
		self._layerOutput = []

		# Create weight arrays
		for (l1,l2) in zip(layerSize[:-1],layerSize[1:]):
			self.weights.append(np.random.normal(scale=0.1, size = (l2, l1+1)))


	def Run(self, input):										# input - rows where each row represents the data 
		inCases = input.shape[0]
		self._layerInput = []
		self._layerOutput = []
		for index  in range(self.layerCount):
			if index == 0:
				layerInput = self.weights[0].dot(np.vstack([input.T, np.ones([1, inCases])]))
			else:
				layerInput = self.weights[index].dot(np.vstack([self._layerOutput[-1], np.ones([1, inCases])]))

			self._layerInput.append(layerInput)
			self._layerOutput.append(self.sgm(layerInput))

		return self._layerOutput[-1].T

	def TrainEpoch(self, input, target, trainingRate = 0.2):
		# This method trains the network for one epoch
		delta = []
		inCases = input.shape[0]
		self.Run(input)

		# Compute deltas
		for index in reversed(range(self.layerCount)):
			if index == self.layerCount-1:
				output_delta = self._layerOutput[index] - target.T
				error = np.sum(output_delta**2)
				delta.append(output_delta * self.sgm(self._layerInput[index], True))
			else:
				delta_pullback = self.weights[index+1].T.dot(delta[-1])
				delta.append(delta_pullback[:-1,:]*self.sgm(self._layerInput[index], True))

		# Compute weight deltas
		for index in range(self.layerCount):
			delta_index = self.layerCount-1-index
			if index == 0:
				layerOutput = np.vstack([input.T, np.ones([1, inCases])])
			else:
				layerOutput = np.vstack([self._layerOutput[index-1], np.ones([1, self._layerOutput[index-1].shape[1]])])

			weightDelta = np.sum(\
								layerOutput[None,:,:].transpose(2,0,1) * delta[delta_index][None,:,:].transpose(2,1,0), \
								axis = 0)

			self.weights[index] -= trainingRate * weightDelta

		return error

	#transfer function
	def sgm(self, x, Derivative=False):
		if not Derivative:
			return 1/(1+np.exp(-x))
		else:
			out = self.sgm(x)
			return out*(1-out) 

if __name__ == "__main__":
	bpn = BackPropagationNetwork((2,2,1))							# (no.ofInputs, no.ofLayers, no.ofOutputs)
	print bpn.shape
	print bpn.weights

	# Training Data
	lvInput = np.array([[0,0],[0,1],[1,0],[1,1]])
	lvTarget = np.array([[0.05],[0.95],[0.95],[0.05]])

	lnMax = 100000
	lnErr = 1e-5
	for i in range(lnMax):
		err = bpn.TrainEpoch(lvInput, lvTarget)
		if i%10000 == 0:
			print "Iteration: ",i," Error: ", err
		if err <= lnErr:
			print "Minimum error at: ", i
			break

	# Testing Data
	lvInput2 = np.array([[0,1],[0,1],[1,0],[1,0],[0,0],[1,1]])
	lvOutput = bpn.Run(lvInput2)
	print "Input: ", lvInput2
	print "Output: ", lvOutput

