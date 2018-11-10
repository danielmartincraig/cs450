###############################################################################
#                            Neural Net Classifier
#                                   CS450
#                                Daniel Craig
###############################################################################

import numpy as np
import random
import math
from pandas import DataFrame, concat, read_csv
from scipy.stats import zscore
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelBinarizer


###############################################################################  
class datasetFromCsv(object):
    """ Represents a dataset; the data is read from a csv file """
    def __init__(self, sep):
        self.filedata = read_csv(self.filename, header = None, names = self.attribute_names, skipinitialspace=True, sep=sep)


###############################################################################  
class IrisData(datasetFromCsv):
    """ Represents the Iris data """
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
    """ Represents the neural net classifier """
    def __init__(self, learning_rate):
        # Don't generate the neuralNet in the constructor, wait until fit() is called 
        self.neuralNetwork = None
        self.activationFunction = np.vectorize(lambda x: 1 / (1 + math.e ** -x))
        self.learning_rate = .3

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
        for layer in self.neuralNetwork.layers:
            # Add the bias node
            data = DataFrame(data).assign(bias=-1)
            # Get the activations
            data = np.dot(data, layer)
            # apply the activation function to the vector
            data = self.activationFunction(data)

        activations = data

        # Feed the error backwards through the neural net and update the weights
        # Start by calculating the error at the output nodes
        # delta_output(k) = (a_k - t_k) * a_k * (1 - a_k)
        errorAtOutputNodes = activations*(1 - activations)*(activations - targets)

        # layer2 = self.neuralNetwork.layers[-1]
        # errorAtOutputNodes.T

        # np.dot(layer2, errorAtOutputNodes.T)

        # Calculate the error at every layer of hidden nodes
        # delta_hidden(j) = a_j * (1 - a_j) \sum_{k=1}^{N} (w_j * delta_output(k))
        # for layer in self.neuralNetwork.layers[::-1]:
        #     pass

        return NNModel(self.neuralNetwork)


###############################################################################  
class NNModel(object):
    """ Represents a trained NN model, used for classification"""
    def __init__(self, neuralNetwork):
        self.neuralNetwork = neuralNetwork

    def predict(self, testing_data):
        return ["No Results!"]


###############################################################################  
class neuralNet(object):
    """ Represents a neural network """
    def __init__(self, numberOfInputNodes, numbersOfNodesInLayers):
        # Handle exceptional cases
        if numberOfInputNodes <= 0:
            raise Exception("Neural network must have positive number of input nodes")
        if len(numbersOfNodesInLayers) == 0:
            raise Exception("Cannot create the neural net without knowledge of how many nodes to put in each of how many layers")
        
        # Store the number of input nodes
        self.numberOfInputNodes = numberOfInputNodes
    
        # Create the first layer randomly.  It will be a matrix of size (numberOfInputNodes X numberOfNodesInFirstLayer)
        numberOfNodesInFirstLayer = numbersOfNodesInLayers[0]
        firstLayer = np.array([[random.uniform(-1,1) for j in range(numberOfNodesInFirstLayer)] for i in range(numberOfInputNodes + 1)])
        self.layers = [firstLayer]

        # Create the other layers, if the they are supposed to exist
        for numberOfNodes in numbersOfNodesInLayers[1:]:
            numberOfInputNodesToNewLayer = len(self.layers[-1][0])
            newLayer = np.array([[random.uniform(-1,1) for j in range(numberOfNodes)] for i in range(numberOfInputNodesToNewLayer + 1)])
            self.layers.append(newLayer)

        # Create a place to store the activations of the nodes in the net
        self.activations = [None for numberOfNodes in numbersOfNodesInLayers]

    @property
    def numberOfLayers(self):
        """ Returns the number of layers in the neural network """
        return len(self.layers)

    # def classify(self, inputs):
    #     # Iterate through the layers in the neural net
    #     for layer in self.layers:
    #         # Add the bias node
    #         inputs = DataFrame(inputs).assign(bias=-1)
    #         # Get the activations
    #         inputs = np.dot(inputs, layer)
    #         # apply the activation function to the vector
    #         inputs = self.activationFunction(inputs)
        
    #     return inputs

###############################################################################
# My neural net in practice
###############################################################################
def main():
    # Get the data
    IrisDataObject = IrisData()
    data = IrisDataObject.data
    targets = IrisDataObject.targets
    label_binarizer = LabelBinarizer()
    label_binarizer.fit(targets['labels'].unique())
    onehot_targets = label_binarizer.transform(targets)

    # Declare the classifier
    classifier = NNClassifier(learning_rate = .2)

    # Declare a list to hold the accuracy scores
    accuracy_list = []
    
    # Create the KFold split object 
    kf = KFold(n_splits=2)
    for train_index, test_index in kf.split(data, onehot_targets):

        # Build the data/target lists
        training_data = data.iloc[train_index] 
        training_targets = onehot_targets[train_index]
        testing_data = data.iloc[test_index]
        testing_targets = onehot_targets[test_index]

        # Build the model
        model = classifier.fit(training_data, training_targets)

        # Predict
        predicted_classes = model.predict(testing_data)
        
        # Add the results to the list of accuracy scores
        # accuracy_list.append(accuracy_score(testing_targets, predicted_classes))


if __name__ == '__main__':
    main()
