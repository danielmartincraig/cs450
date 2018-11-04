###############################################################################
#                           Neural Net Milestone 1
#
#                                Daniel Craig
###############################################################################

import numpy as np
import random
from pandas import DataFrame, concat, read_csv

# Build a classifier with an arbitrary number of layers, and an arbitrary number of nodes in each layer.
#     For example: 2 Layers (1 hidden, 1 output), with 4 nodes in the hidden layer, and 3 output nodes.
#     Or 3 Layers (2 hidden, 1 output), with 2 nodes in the first, 3 in the second, 1 in the third.
#     Or 2 Layers (1 hidden, 1 output) with 8 nodes in the hidden layer, and 2 in the output.
#     Etc.
# The number of weights for the first layer should be determined by the number of input attributes.
# Biases are present at every layer.
# You should be able to complete the feed-forward portion of the algorithm, looping through each node of each layer to produce values at the output layer.
# For your activation function, use the sigmoid function: f(x) = 1 / (1 + e^-x)
# You should be able to classify an instance using your network. Please note that at this point, you will not have implemented any weight updates, so your network will essentially be randomly computing an answer, but it should be able to classify a given instance.
# Using the classification described, you should be able to classify a complete dataset and calculate the accuracy. (E.g., try it on the iris dataset)

class datasetFromCsv(object):
    """ This class represents a dataset; the data is read from a csv file """
    def __init__(self, sep):
        self.filedata = read_csv(self.filename, header = None, names = self.attribute_names, skipinitialspace=True, sep=sep)


class ChessData(datasetFromCsv):
    """ This class represents the Chess data """
    def __init__(self):
        super(ChessData, self).__init__(sep=',')

    @property
    def data(self):
        return DataFrame({
            "WKingFile": self.filedata["WKingFile"].astype('category').cat.codes,
            "WKingRank": self.filedata["WKingRank"].astype('category').cat.codes,
            "WRookFile": self.filedata["WRookFile"].astype('category').cat.codes,
            "WRookRank": self.filedata["WRookRank"].astype('category').cat.codes,
            "BKingFile": self.filedata["BKingFile"].astype('category').cat.codes,
            "BKingRank": self.filedata["BKingRank"].astype('category').cat.codes,
        })

    @property
    def filename(self):
        return "/home/daniel/Repos/cs450/resources/chess/archive.ics.uci.edu/ml/machine-learning-databases/chess/king-rook-vs-king/krkopt.data"

    @property
    def attribute_names(self):
        return ["WKingFile",
                "WKingRank",
                "WRookFile",
                "WRookRank",
                "BKingFile",
                "BKingRank",
                "OptimalDepthOfWin"]

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
        
        newLayer = np.array([[random.uniform(-1,1) for j in range(numberOfNodesInFirstLayer)] for i in range(numberOfInputNodes + 1)])
        self.layers = [newLayer]

    def addLayer(self, numberOfNodes):
        numberOfInputNodesToNewLayer = len(self.layers[-1][0])
        newLayer = np.array([[random.uniform(-1,1) for j in range(numberOfNodes)] for i in range(numberOfInputNodesToNewLayer + 1)])

        self.layers.append(newLayer)

    @property
    def numberOfLayers(self):
        return len(self.layers)

    def classify(self, inputVector):
        for layer in self.layers:
            inputVector.append(-1)
            inputVector = list(np.dot(inputVector, layer))
        return inputVector

###############################################################################
# getSingleLayerPerceptron 
###############################################################################  
#def getSingleLayerPerceptron(numberOfInputs, numberOfNodes):
#    neuralNet = np.array

#    randomFloat = lambda: random.uniform(-1,1)
#    perceptron = neuralNet([[randomFloat() for j in range(numberOfNodes)] for i in range(numberOfInputs + 1)])
#    # perceptron = np.append(perceptron, [np.array([-1 for i in range(numberOfNodes)])], axis=0)
#    return perceptron

###############################################################################
# My neural net in practice
###############################################################################
def main():
    ChessDataObject = ChessData()
    vectors = [vector for vector in ChessDataObject.data.values.tolist()]

    myNet = neuralNet(numberOfInputNodes=6, numberOfNodesInFirstLayer=6)
    myNet.addLayer(numberOfNodes=2)
    myNet.addLayer(numberOfNodes=5)

    labelActivations = lambda activation: np.where(activation>0,1,0)
    for vector in vectors:
        print labelActivations(myNet.classify(vector))

if __name__ == '__main__':
    main()
