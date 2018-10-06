###############################################################################
# An experiment on the following datasets:
#     UCI: Car Evaluation
#     Autism Spectrum Disorder 
#     Automobile MPG 
#
# For each dataset, read it in, handle missing values, text data, etc.
# Then use a kNN classifier with varying values for k.
#
# For each dataset and configuration, use k-fold cross validation to compare 
# your results.
###############################################################################

import pandas as pd
from pandas import DataFrame, Series
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

class datasetFromCsv(object):
    """ This class represents a dataset; the data is read from a csv file """
    def __init__(self):
        self.filedata = pd.read_csv(self.filename, skipinitialspace = True, header = None, names = self.attributeNames)

    @property
    def data(self):
        return pd.DataFrame({attributeName: self.filedata[attributeName].astype('category').cat.codes for attributeName in self.attributeNames[:-1]})
        
    @property
    def target_names(self):
        return sorted(list(set(self.filedata[self.attributeNames[-1]])))

    @property
    def targets(self):
        return self.filedata[self.attributeNames[-1]].astype('category').cat.codes


class uciCarEvaluation(datasetFromCsv):
    """ This class represents the UCI data """
    def __init__(self):
        super(uciCarEvaluation, self).__init__()

    @property
    def filename(self):
        return "/home/daniel/Repos/cs450/resources/uciCarEvaluation/archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"

    @property
    def attributeNames(self):
        return ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]


class autismData(datasetFromCsv):
    """ This class represents the data related to Autism """
    def __init__(self):
        super(autismData, self).__init__()

    @property
    def filename(self):
        return "/home/daniel/Repos/cs450/resources/autismData/archive.ics.uci.edu/ml/machine-learning-databases/00426/Autism-Adult-Data.arff"

    @property
    def attributeNames(self):
        return ["A1_Score", "A2_Score", "A3_Score", "A4_Score", "A5_Score", "A6_Score", "A7_Score", "A8_Score", "A9_Score", "A10_Score", "age", "gender", "ethnicity", "jundice", "austim", "contry_of_res", "used_app_before", "result", "age_desc", "relation", "Class/ASD"]


class automobileMPG(datasetFromCsv):
    """ This class represents the MPG data """
    def __init__(self):
        super(automobileMPG, self).__init__()

    @property
    def filename(self):
        return "/home/daniel/Repos/cs450/resources/automobileMPG/archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"

    @property
    def attributeNames(self):
        return ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model year", "origin", "car name"] 


def categorizeItems(column):
    """ This function updates columns with numeric category values """
    categories = sorted(list(set(column)))

    nameMap = {}
    for i, name in enumerate(categories):
        nameMap[name] = i

    return [nameMap[item] for item in column]


# Minimum Standard Requirements
#     Read data from text files
#     Appropriately handle non-numeric data.
#     Appropriately handle missing data.
#     Use of k-Fold Cross Validation
#     Basic experimentation on the provided datasets.

def main():
    # # # Part 1: UCI Car Evaluation Experiment
    carEvaluationDataObject = uciCarEvaluation()
    data = carEvaluationDataObject.data
    targets = carEvaluationDataObject.targets
    target_names = carEvaluationDataObject.target_names

    training_data, testing_data, training_targets, testing_targets = train_test_split(data, targets, test_size=0.33)

    k = 3

    classifier = KNeighborsClassifier(n_neighbors=k)
    model = classifier.fit(training_data, training_targets)

    predicted_classes = model.predict(testing_data)

    accuracy = accuracy_score(testing_targets, predicted_classes)
    print "The KNN model predicted cars classes with an accuracy of", accuracy


    # # # Part 2: Experiment on Data Related to Autism
    autismDataObject = autismData()
    data = autismDataObject.data
    targets = autismDataObject.targets
    target_names = autismDataObject.target_names

    training_data, testing_data, training_targets, testing_targets = train_test_split(data, targets, test_size=0.33)

    k = 3

    classifier = KNeighborsClassifier(n_neighbors=k)
    model = classifier.fit(training_data, training_targets)

    predicted_classes = model.predict(testing_data)

    accuracy = accuracy_score(testing_targets, predicted_classes)
    print "The KNN model predicted autism classes with an accuracy of", accuracy

    # # # Part 3: Auto MPG Experiment
    automobileMPGDataObject = automobileMPG()
    data = automobileMPGDataObject.data
    targets = automobileMPGDataObject.targets
    target_names = automobileMPGDataObject.target_names

    training_data, testing_data, training_targets, testing_targets = train_test_split(data, targets, test_size=0.33)

    k = 3

    classifier = KNeighborsClassifier(n_neighbors=k)
    model = classifier.fit(training_data, training_targets)

    predicted_classes = model.predict(testing_data)

    accuracy = accuracy_score(testing_targets, predicted_classes)
    print "The KNN model predicted car classes with an accuracy of", accuracy

if __name__ == "__main__":
    main()
