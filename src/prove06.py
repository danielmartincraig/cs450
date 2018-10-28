###############################################################################
#                           Neural Net Milestone 1
#
#                                Daniel Craig
###############################################################################

import numpy as np
import random

#  Create some type of mechanism to represent your nodes (i.e. neurons) and their
#  associated weights.
neuralNet = np.array
inputVector = np.array

#  Provide a way to create a single layer of nodes of any number (this should be
#  easily specified via a parameter).

#  Account for a bias input.

#  Be able to take input from a dataset instance (with an arbitrary number of
#  attributes) and have each node produce an output (i.e., 0 or 1) according to
#  its weights.

#  Be able to load and process at least one dataset of your choice.

#  You should appropriately normalize the data set.

###############################################################################
# This is addLayer, it might come in handy I guess
###############################################################################
# def addLayer(net, numberOfNodes):
#     randomFloat = lambda: random.uniform(-1,1)
#     nodesInPreviousLayer = len(net[-1])
#     newLayer =  np.array([[randomFloat() for i in range(numberOfNodes)] for j in range(nodesInPreviousLayer)])
#     np.append(net, newLayer, axis=0)


###############################################################################
# My neural net in practice
###############################################################################  

myInputVector = inputVector([.5,.5,.5])
myNet = neuralNet([[1,1,1,1,1],[2,2,2,2,2],[3,3,3,3,3]])

addLayer(myNet, numberOfNodes=5)