import numpy as np
import matplotlib.pyplot as plt

class Neural_Network(object):
    """docstring for Neural_Network."""
    def __init__(self):
        #Define HyperParameters
        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerSize = 3

        #Weights (Parameters)
        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)

    def forward(self,X):
        #Propagate inputes through network.
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3)
        return yHat

    def sigmoid(self,z):
        #Apply sigmoid activation function
        return 1./(1.+np.exp(-z))

    def sigmoidPrime(self,z):
        #Derivative of sigmoid function
        return np.exp(-z)/((1.+np.exp(-z))**2)

    def costFunction(self,X,y):
        self.yHat=self.forward(X)
        J=0.5*sum((y-yHat)**2)
        return J

    def costFunctionPrime(self,X,y):
        #Compute derivative with respect to W1 and W2
        self.yHat = self.forward(X)

        delta3 = np.multiply(-(y-self.yHat),self.sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.T,delta3)

        delta2 = np.dot(delta3,self.W2.T)*self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(X.T, delta2)

        return dJW1, dJdW2


def computeNumericalGradient(N, X, y):
        paramsInitial = N.getParams()
        numgrad = np.zeros(paramsInitial.shape)
        perturb = np.zeros(paramsInitial.shape)
        e = 1e-4

        for p in range(len(paramsInitial)):
            #Set perturbation vector
            perturb[p] = e
            N.setParams(paramsInitial + perturb)
            loss2 = N.costFunction(X, y)

            N.setParams(paramsInitial - perturb)
            loss1 = N.costFunction(X, y)

            #Compute Numerical Gradient
            numgrad[p] = (loss2 - loss1) / (2*e)

            #Return the value we changed to zero:
            perturb[p] = 0

        #Return Params to original value:
        N.setParams(paramsInitial)

        return numgrad 

# testInput = np.arange(-6,6,0.01)
# plt.plot(testInput,sigmoid(testInput),linewidth=2)
# plt.grid(1)
# plt.show()

#Training Network == Minimizing a cost function

X=np.array(([3,5],[5,1],[10,2]), dtype =float)
y=np.array(([75],[82],[93]), dtype = float)

#Normalization of data:

X=X/np.amax(X, axis=0)
y=y/100.

NN = Neural_Network()

cost1 = NN.costFunction(X,y)
