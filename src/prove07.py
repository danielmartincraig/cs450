###############################################################################
#                           Neural Net Milestone 1
#
#                                Daniel Craig
###############################################################################

import numpy as np
import random
import math
from pandas import DataFrame, concat, read_csv
from scipy.stats import zscore

# Build a classifier with an arbitrary number of layers, and an arbitrary number of nodes in each layer.
#     For example: 2 Layers (1 hidden, 1 output), with 4 nodes in the hidden layer, and 3 output nodes.
#     Or 3 Layers (2 hidden, 1 output), with 2 nodes in the first, 3 in the second, 1 in the third.
#     Or 2 Layers (1 hidden, 1 output) with 8 nodes in the hidden layer, and 2 in the output.
#     Etc.
# The number of weights for the first layer should be determined by the number of input attributes.
# Biases are present at every layer.
# You should be able to complete the feed-forward portion of the algorithm, looping through each node of each layer to produce values at the output layer.
# For your activation function, use the sigmoid function: f(x) = 1 / (1 + e^-x)
# You should be able to classify an instance using your network. Please note
# that at this point, you will not have implemented any weight updates, so your
# network will essentially be randomly computing an answer, but it should be able
# to classify a given instance.
# Using the classification described, you should be able to classify a complete dataset and calculate the accuracy. (E.g., try it on the iris dataset)

class datasetFromCsv(object):
    """ This class represents a dataset; the data is read from a csv file """
    def __init__(self, sep):
        self.filedata = read_csv(self.filename, header = None, names = self.attribute_names, skipinitialspace=True, sep=sep)


class IrisData(datasetFromCsv):
    """ This class represents the Iris data """
    def __init__(self):
        super(IrisData, self).__init__(sep=',')

    @property
    def data(self):
        return DataFrame({
            "sepalLength": self.filedata["sepalLength"],
            "sepalWidth": self.filedata["sepalWidth"],
            "petalLength": self.filedata["petalLength"],
            "petalWidth": self.filedata["petalWidth"],
        }).apply(zscore)

    @property
    def filename(self):
        return "/home/daniel/Repos/cs450/resources/iris/archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

    @property
    def attribute_names(self):
        return ["sepalLength",
                 "sepalWidth",
                 "petalLength",
                 "petalWidth",
                 "class"]

    @property
    def target_names(self):
        return sorted(list(set(self.filedata[self.attribute_names[-1]])))

    @property
    def targets(self):
        return DataFrame({'labels': self.filedata[self.attribute_names[-1]].astype('category').cat.codes})

###############################################################################
# This class defines my neural network
###############################################################################  
class neuralNet(object):
    def __init__(self, numberOfInputNodes, numberOfNodesInFirstLayer):
        self.numberOfInputNodes = numberOfInputNodes
        self.activationFunction = np.vectorize(lambda x: 1 / (1 + math.e ** -x))
        
        newLayer = np.array([[random.uniform(-1,1) for j in range(numberOfNodesInFirstLayer)] for i in range(numberOfInputNodes + 1)])
        self.layers = [newLayer]

    def addLayer(self, numberOfNodes):
        numberOfInputNodesToNewLayer = len(self.layers[-1][0])
        newLayer = np.array([[random.uniform(-1,1) for j in range(numberOfNodes)] for i in range(numberOfInputNodesToNewLayer + 1)])

        self.layers.append(newLayer)

    @property
    def numberOfLayers(self):
        return len(self.layers)

    def classify(self, inputs):
        # Iterate through the layers in the neural net
        for layer in self.layers:
            # Add the bias node
            inputs = DataFrame(inputs).assign(bias=-1)
            # Get the activations
            inputs = np.dot(inputs, layer)
            # apply the activation function to the vector
            inputs = self.activationFunction(inputs)
        
        return inputs

###############################################################################
# My neural net in practice
###############################################################################
def main():
    IrisDataObject = IrisData()
    data = IrisDataObject.data
    targets = IrisDataObject.targets

    myNet = neuralNet(numberOfInputNodes=4, numberOfNodesInFirstLayer=6)
    myNet.addLayer(numberOfNodes=2)
    myNet.addLayer(numberOfNodes=5)

    activations = myNet.classify(data)
    labelActivations = lambda activation: np.where(activation>0.5,1,0)
    for row in activations:
        print labelActivations(row)

if __name__ == '__main__':
    main()
