# An experiment on the following datasets:
#     UCI: Car Evaluation
#     Autism Spectrum Disorder 
#     Automobile MPG 
#
# For each dataset, read it in, handle missing values, text data, etc.
# Then use a kNN classifier with varying values for k.
#
# For each dataset and configuration, use k-fold cross validation to compare your results.

import pandas as pd


class datasetFromCsv(object):
    """ This class represents a dataset; the data is read from a csv file """
    def __init__(self):
        self.filedata = pd.read_csv(self.filename, skipinitialspace = True, header = None, names = self.attributeNames)
        self.data = []
        self.targets = []
        self.target_names = []


class uciCarEvaluation(datasetFromCsv):
    """ This class represents the UCI data """
    def __init__(self):
        self.filename = "/home/daniel/Repos/cs450/resources/uciCarEvaluation/archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
        self.attributeNames = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
        super(uciCarEvaluation, self).__init__()
        

class autismData(datasetFromCsv):
    """ This class represents the data related to Autism """
    def __init__(self):
        self.attributeNames = []
        self.filename = "/home/daniel/Repos/cs450/resources/autismData/archive.ics.uci.edu/ml/machine-learning-databases/00426/Autism-Adult-Data.arff"


class automobileMPG(datasetFromCsv):
    """ This class represents the MPG data """
    def __init__(self):
        self.attributeNames = []
        self.filename = "/home/daniel/Repos/cs450/resources/automobileMPG/archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"


# Minimum Standard Requirements
#     Read data from text files
#     Appropriately handle non-numeric data.
#     Appropriately handle missing data.
#     Use of k-Fold Cross Validation
#     Basic experimentation on the provided datasets.

def main():
    # Part 1: UCI Car Evaluation Experiment
    carData = uciCarEvaluation(); 

    # Part 2: Experiment on Data Related to Autism

    # Part 3: Auto MPG Experiment


if __name__ == "__main__":
    main()