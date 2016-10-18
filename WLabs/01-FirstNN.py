import numpy as np
import matplotlib.pyplot as plt

class Neural_Network(object):
    """docstring for Neural_Network."""
    def __init__(self):
        #Define HyperParameters
        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerSize = 3

    def forward(self,X):
        #Propagate inputes through network.
        return 1

def sigmoid(z):
    #Apply sigmoid activation function
    return 1./(1.+np.exp(-z))

testInput = np.arange(-6,6,0.01)
plt.plot(testInput,sigmoid(testInput),linewidth=2)
plt.grid(1)
plt.show()
