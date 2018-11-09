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



###############################################################################  
class datasetFromCsv(object):
    """ This class represents a dataset; the data is read from a csv file """
    def __init__(self, sep):
        self.filedata = read_csv(self.filename, header = None, names = self.attribute_names, skipinitialspace=True, sep=sep)


###############################################################################  
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
class NNClassifier(object):
    """This class represents the neural net classifier"""
    def __init__(self):
        # Don't generate the neuralNet in the constructor, wait until fit() is called 
        self.neuralNetwork = None

    def fit(self, data, targets):
        # Handle exceptional cases
        if data.shape[0] == 0 or targets.shape[0] == 0:
            raise Exception("The training data and targets passed to fit() cannot be null.")
        if data.shape[0] != targets.shape[0]:
            raise Exception("The number of training records did not match the number of training targets.")

        # get the number of attributes - this equals the number of inputs to the first layer (not including bias)
        numberOfInputNodes = data.shape[1]
        # Specify how many nodes are in how many layers
        numberOfNodesInLayers = [4,3,3]
        
        self.neuralNetwork = neuralNet(numberOfInputNodes=numberOfInputNodes, numbersOfNodesInLayers=numberOfNodesInLayers)



        # Feed the inputs forward
        # Compute the error at the output nodes
        # Feed the error backwards through the neural net


        # for layer in self.layers:
            
        #     # Add the bias node and calculate the activations
        #     inputs = DataFrame(inputs).assign(bias=-1)
        #     inputs = np.dot(inputs, layer)
            
        #     # "the inputs and the first-layer weights (here labelled as v) are used to decide
        #     # whether the hidden nodes fire or not. The activation function g(x) is the sigmoid
        #     # function given in Equation (4.2) above"
        #     inputs = self.activationFunction(inputs)


        #     # "the outputs of these neurons and the second-layer weights (labelled as w) are
        #     # used to decide if the output neurons fire or not"

        #         # apply the activation function to the vector
            
        #     return inputs



    def predict(self, testing_data):
        return ["No Results!"]

###############################################################################
# This class defines my neural network
###############################################################################  
class neuralNet(object):
    def __init__(self, numberOfInputNodes, numbersOfNodesInLayers):
        if numberOfInputNodes <= 0:
            raise Exception("Neural network must have positive number of input nodes")
        if len(numbersOfNodesInLayers) == 0:
            raise Exception("Cannot create the neural net without knowledge of how many nodes to put in each of how many layers")
        
        self.numberOfInputNodes = numberOfInputNodes
        # self.activationFunction = np.vectorize(lambda x: 1 / (1 + math.e ** -x))

        numberOfNodesInFirstLayer = numbersOfNodesInLayers[0]
        firstLayer = np.array([[random.uniform(-1,1) for j in range(numberOfNodesInFirstLayer)] for i in range(numberOfInputNodes + 1)])
        self.layers = [firstLayer]

        for numberOfNodes in numbersOfNodesInLayers[1:]:
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
    # Get the data
    IrisDataObject = IrisData()
    data = IrisDataObject.data
    targets = IrisDataObject.targets

    # Declare the classifier
    classifier = NNClassifier()
    model = classifier.fit(data, targets)

    # predicted_targets = model.predict(testing_data)


if __name__ == '__main__':
    main()
