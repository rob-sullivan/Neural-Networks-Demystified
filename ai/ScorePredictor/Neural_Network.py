"""Supervised Regression Problem"""
import numpy as np

"""Training Data"""
#Input Data = clicks on website and emails opened
X = np.array(([3,5],[5,1], [10,2]), dtype=float)

#Output data = lead generation score
y = np.array(([75],[82], [93]), dtype=float)

"""Testing Data"""
#(8,3) = ?

"""
Using Artificial Neural Networks
Other ways are as follows:
	1. Support Vector Machines
	2. Gaussion Process
	3. Classification + Regression Trees
	4. Random Forests
	5. Etc.
"""

"""Now need to account for units of data. X is in clicks y is in 1 of 100%"""
#Scale data (All data is positive)

y = y/100 # Max lead score is 100
#output layer will be called y-hat because it is an estimate of y. Not y.
class Neural_Network(object):
	def __init__(self):
		#Define HyperParameters
		self.inputLayerSize = 2
		self.outputLayerSize = 1
		self.hiddenLayerSize = 3
		
		#Weights (Parameters)
		self.W1 = np.random.rand(self.inputLayerSize, \
									self.hiddenLayerSize)
		self.W2 = np.random.rand(self.hiddenLayerSize, \
									self.outputLayerSize)		
	def forward(self, X):
		#Propagate inputs through network
		self.z2 = np.dot(X, self.W1)
		self.a2 = self.sigmoid(self.z2)
		self.z3 = np.dot(self.a2, self.W2)
		yHat = self.sigmoid(self.z3)
		return yHat
		
	def sigmoid(self, z):
		#Apply sigmoid activation function
		return 1/(1+np.exp(-z))

	"""		
	#Initial test before Gradient Decent added
	NN = Neural_Network()
	yHat = NN.forward(X)
	print("\Website Clicks and Mail opened:\n\n %s" % X)
	print("\nPredicted Lead Score:\n\n %s" % yHat)
	print("\nActual Lead Score:\n\n %s" % y)
	"""

	"""
	Bad predictions made because we haven't trained the network.
	If we use a brute force method we get trapped in the curse of dementionality
	We will define a cost fuction J
	J = SUM(0.5 * (y-f(f(X*W1))*W2))^2

	We will diferenciate J with respect to W. 
	We will only do this for one W at a time so will use Sigma
	Sigma J / Sigma W
	This will let us do Gradient Decent

	Gradient Decent will speed up our network.
	What would have taken 10^27 cost evaluations for our 3 weights using brute force,
	will now take less than 100 evaluations.

	We took into consideration a non-convex issue. Where our cost goes down then back up.
	We can get stuck in a local min when we want a global min if we don't sort this out.

	That's why we square the sum of the errors in the J formula.
	This allows us to take advantage of convex nature of quadratic equations

	We could have used stochastic gradient decent where we look at one error at a time,
	but we want to do Gradient Decent, batch style to speed up our network.
	"""
	#Back Propagation

	"""
	We are going to seperate our Sigma J / Sigma W into Sigma J / Sigma W1 and Sigma J / Sigma W2
	We did the same thing with our weights due to them being different matrices
	This makes sigma j/ sigma w the same size as our weights

	Sigma J / Sigma W2 = Sigma(SUM(0.5 * (y-yHat)^2)) / Sigma W2

	We now use sum rule in differentation
	Sigma J / Sigma W2 = (y-yHat)
	Now we use the chain rule
	Sigma J / Sigma W2 = 1(y-yHat) * Sigma yHat / Sigma z3 * Sigma yHat / Sigma W2
	We use differentation on our sigmoid activation function
	F'(z) = e^-z/ (1+e^-2)^2
	"""
	def sigmoidPrime(self, z):
		#Derivative of sigmoid function
		return np.exp(-z)/((1+np.exp(-z))**2)

	"""
	#testing sigmoid prime functioin vs sigmoid function (matplotlib needed)
	testValues = np.arange(-5,5,0.01)
	plot(testValues, sigmoid(testValues), linewidth=2)
	plot(testValues, sigmoidPrime(testValues), linewidth=2)
	grid(1)
	legend(['sigmoid', 'sigmoidPrime'])

	Delta^3 will be the back propagation error
	We transpose a^2 and matrix multiply Delta^3
	"""

	def costFunction(self, X, y):
		#Compute cost for given X,y, use weights already stored in class.
		self.yHat = self.forward(X)
		J = 0.5*sum((y-self.yHat)**2)
		return J

	def costFunctionPrime(self, X, y):
		#Compute derivative with respect to W1 and W2
		self.yHat = self.forward(X)
		delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
		dJdW2 = np.dot(self.a2.T, delta3)
		
		#if we want to make a deeper network we stack a bunch of these operations together
		delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
		dJdW1 = np.dot(X.T, delta2)
		
		return dJdW1, dJdW2
	

	
#Test after Gradient Decent added
NN = Neural_Network()
cost1 = NN.costFunction(X, y)
dJdW1, dJdW2 = NN.costFunctionPrime(X,y)
print("\nWhich way is up?:\n\n %s" % dJdW1)
print("\n%s" % dJdW2)

scalar = 3
NN.W1 = NN.W1 + scalar*dJdW1
NN.W2 = NN.W2 + scalar*dJdW2
cost2 = NN.costFunction(X,y)

print("\nCosts 1 and Cost 2:\n\n %s" % cost1)
print("\n%s" % cost2)

dJdW1, dJdW2 = NN.costFunctionPrime(X,y)
NN.W1 = NN.W1 + scalar*dJdW1
NN.W2 = NN.W2 + scalar*dJdW2
cost3 = NN.costFunction(X,y)

print("\nCosts 1 and Cost 2:\n\n %s" % cost2)
print("\n%s" % cost3)

