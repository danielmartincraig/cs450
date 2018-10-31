###############################################################################
#                           Neural Net Milestone 1
#
#                                Daniel Craig
###############################################################################

import numpy as np
import random
from pandas import DataFrame, concat, read_csv

#  Create some type of mechanism to represent your nodes (i.e. neurons) and their
#  associated weights.

#  Provide a way to create a single layer of nodes of any number (this should be
#  easily specified via a parameter).

#  Account for a bias input.

#  Be able to take input from a dataset instance (with an arbitrary number of
#  attributes) and have each node produce an output (i.e., 0 or 1) according to
#  its weights.

#  Be able to load and process at least one dataset of your choice.

#  You should appropriately normalize the data set.

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


def getSingleLayerPerceptron(numberOfInputs, numberOfNodes):
    neuralNet = np.array

    randomFloat = lambda: random.uniform(-1,1)
    perceptron = neuralNet([[randomFloat() for j in range(numberOfNodes)] for i in range(numberOfInputs)])
    # perceptron = np.append(perceptron, [np.array([-1 for i in range(numberOfNodes)])], axis=0)
    return perceptron

###############################################################################
# My neural net in practice
###############################################################################  
def main():
    inputVector = np.array

    ChessDataObject = ChessData()


    myNet = getSingleLayerPerceptron(6,5)

    activations = [np.dot(vector, myNet) for vector in ChessDataObject.data.values.tolist()]
    labelActivations = lambda activation: np.where(activation>0,1,0)
    for row in activations:
        print labelActivations(row)

if __name__ == '__main__':
    main()