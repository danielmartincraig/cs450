###############################################################################
#                            Neural Net Classifier
#                                   CS450
#                                Daniel Craig
###############################################################################

import numpy as np
import random, math
from pandas import DataFrame, concat, read_csv
from scipy.stats import zscore
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelBinarizer
from collections import deque
from itertools import islice


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
        self.learning_rate = learning_rate

    def fit(self, training_data, targets):
        # Handle exceptional cases
        if training_data.shape[0] == 0 or targets.shape[0] == 0:
            raise Exception("The training data and targets passed to fit() cannot be null.")
        if training_data.shape[0] != targets.shape[0]:
            raise Exception("The number of training records did not match the number of training targets.")

        # get the number of attributes - this equals the number of inputs to the first layer (not including bias)
        numberOfInputNodes = training_data.shape[1]
        # Specify how many nodes are in how many layers
        numberOfNodesInLayers = [4,5,3]
        
        self.neuralNetwork = neuralNet(numberOfInputNodes=numberOfInputNodes, numbersOfNodesInLayers=numberOfNodesInLayers)

        # Find the activations of the first layer
        # Start by adding the bias to the training data
        training_data = DataFrame(training_data).assign(bias=-1)
        # Then calculate the output
        output = np.dot(training_data, self.neuralNetwork.layers[0])
        # Apply the activation function to the output
        activations = self.activationFunction(output) 
        # Then store the activations as a list in the neural network
        self.neuralNetwork.activations = [activations]

        # Find the activations of the additional layers
        for layer in self.neuralNetwork.layers[1:]:
            # Get the input to the current layer from the previous layer
            inputToLayer = self.neuralNetwork.activations[-1]
            # Create the bias column and add it to the input
            biasColumn = np.full(shape=(self.neuralNetwork.activations[-1].shape[0],1), fill_value=-1)
            inputToLayer = np.append(inputToLayer, biasColumn, 1) # Make sure that this doesn't change the activations in the neural net itself
            # Calculate the output
            output = np.dot(inputToLayer, layer)
            # Apply the activation function
            activations = self.activationFunction(output)
            self.neuralNetwork.activations.append(activations)

        # Calculate the error at the output nodes and add it to the list of errors.
        # delta_output(k) = (a_k - t_k) * a_k * (1 - a_k)
        errorAtOutputNodes = activations * (1 - activations) * (activations - targets)
        self.neuralNetwork.errors = deque([errorAtOutputNodes])

        # Calculate the error at each of the hidden layers and prepend it to the list of errors
        for layer, activations in zip(self.neuralNetwork.layers[::-1], self.neuralNetwork.activations[-2::-1]):
            biasColumn = np.full(shape=(activations.shape[0],1), fill_value=-1)
            augmentedActivations = np.append(activations, biasColumn, axis=1)

            errors = augmentedActivations * (1 - augmentedActivations) * np.dot(layer, self.neuralNetwork.errors[-1].T)
            self.neuralNetwork.errors.appendleft(errors.T)

        # Update the weights 
        for errors, activations, layer in zip(self.neuralNetwork.activations[-2::-1], islice(reversed(self.neuralNetwork.errors), None, None), self.neuralNetwork.layers[::-1]):
            biasColumn = np.full(shape=(activations.shape[0],1), fill_value=-1)
            augmentedActivations = np.append(activations, biasColumn, axis=1)

            meanColumnErrors = np.mean(errors, axis=0)
            meanColumnActivations = np.mean(augmentedActivations, axis=0)
            delta = np.array([meanColumnErrors * activation for activation in meanColumnActivations])
            learningRateTimesDelta = self.learning_rate * delta
            
            layer -= learningRateTimesDelta
            print "Grasshopper!"

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
        self.activations = None
        self.errors = None

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
