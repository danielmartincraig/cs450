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
import numpy as np
from pandas import DataFrame, Series
from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from scipy.stats import zscore

class datasetFromCsv(object):
    """ This class represents a dataset; the data is read from a csv file """
    def __init__(self, sep):
        self.filedata = pd.read_csv(self.filename, header = None, names = self.attributeNames, sep=sep)


class uciCarEvaluation(datasetFromCsv):
    """ This class represents the UCI data """
    def __init__(self):
        super(uciCarEvaluation, self).__init__(sep=', ')

    @property
    def data(self):
        return pd.DataFrame({
            "buying": self.filedata["buying"].astype('category').cat.codes,
            "maint": self.filedata["maint"].astype('category').cat.codes,
            "doors": self.filedata["doors"].astype('category').cat.codes,
            "persons": self.filedata["persons"].astype('category').cat.codes,
            "lug_boot": self.filedata["lug_boot"].astype('category').cat.codes,
            "safety": self.filedata["safety"].astype('category').cat.codes
        })

    @property
    def filename(self):
        return "/home/daniel/Repos/cs450/resources/uciCarEvaluation/archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"

    @property
    def attributeNames(self):
        return ["buying", 
                "maint", 
                "doors", 
                "persons", 
                "lug_boot", 
                "safety", 
                "class"]

    @property
    def target_names(self):
        return sorted(list(set(self.filedata[self.attributeNames[-1]])))

    @property
    def targets(self):
        return self.filedata[self.attributeNames[-1]].astype('category').cat.codes


class autismData(datasetFromCsv):
    """ This class represents the data related to Autism """
    def __init__(self):
        super(autismData, self).__init__(sep=',')
        self.filedata = self.filedata.replace('?', np.nan)
        self.filedata = self.filedata.dropna(axis="rows")

    @property
    def data(self):
        return pd.DataFrame({
            "A1_Score": self.filedata["A1_Score"].astype('bool'),  
            "A2_Score": self.filedata["A2_Score"].astype('bool'),
            "A3_Score": self.filedata["A3_Score"].astype('bool'),
            "A4_Score": self.filedata["A4_Score"].astype('bool'),
            "A5_Score": self.filedata["A5_Score"].astype('bool'),
            "A6_Score": self.filedata["A6_Score"].astype('bool'),
            "A7_Score": self.filedata["A7_Score"].astype('bool'),
            "A8_Score": self.filedata["A8_Score"].astype('bool'),
            "A9_Score": self.filedata["A9_Score"].astype('bool'),
            "A10_Score": self.filedata["A10_Score"].astype('bool'),
            "age": zscore(self.filedata["age"].astype('float')),
            "gender": self.filedata["gender"].astype('category').cat.codes,
            "ethnicity": self.filedata["ethnicity"].astype('category').cat.codes,
            "jundice": self.filedata["jundice"].astype('category').cat.codes,    
            "austim": self.filedata["austim"].astype('category').cat.codes,
            "country_of_res": self.filedata["country_of_res"].astype('category').cat.codes,
            "used_app_before": self.filedata["used_app_before"].astype('category').cat.codes,
            "result": self.filedata["result"].astype('category').cat.codes,
            "age_desc": self.filedata["age_desc"].astype('category').cat.codes,
            "relation": self.filedata["relation"].astype('category').cat.codes,
        })

    @property
    def filename(self):
        return "/home/daniel/Repos/cs450/resources/autismData/archive.ics.uci.edu/ml/machine-learning-databases/00426/Autism-Adult-Data.arff"

    @property
    def attributeNames(self):
        return ["A1_Score",
                "A2_Score",
                "A3_Score", 
                "A4_Score", 
                "A5_Score", 
                "A6_Score", 
                "A7_Score", 
                "A8_Score", 
                "A9_Score", 
                "A10_Score", 
                "age", 
                "gender", 
                "ethnicity", 
                "jundice", 
                "austim", 
                "country_of_res", 
                "used_app_before", 
                "result", 
                "age_desc", 
                "relation", 
                "Class/ASD"]

    @property
    def target_names(self):
        return sorted(list(set(self.filedata[self.attributeNames[-1]])))

    @property
    def targets(self):
        return self.filedata[self.attributeNames[-1]].astype('category').cat.codes


class automobileMPG(datasetFromCsv):
    """ This class represents the MPG data """
    def __init__(self):
        super(automobileMPG, self).__init__(sep='\s+')
        self.filedata = self.filedata.replace('?', np.nan)
        self.filedata = self.filedata.dropna(axis="rows")
        # pass

    @property
    def data(self):
        return pd.DataFrame({
            "cylinders": self.filedata["cylinders"].astype('category').cat.codes,
            "displacement": zscore(self.filedata["displacement"].astype('float')),
            "horsepower": zscore(self.filedata["horsepower"].astype('float')),    
            "weight": zscore(self.filedata["weight"].astype('float')),
            "acceleration": zscore(self.filedata["acceleration"].astype('float')),
            "model_year": self.filedata["model_year"].astype('category').cat.codes,
            "origin": self.filedata["origin"].astype('category').cat.codes,
            "car_name": self.filedata["car_name"].astype('category').cat.codes
        })

    @property
    def filename(self):
        return "/home/daniel/Repos/cs450/resources/automobileMPG/archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"

    @property
    def attributeNames(self):
        return ["mpg", 
                "cylinders", 
                "displacement", 
                "horsepower", 
                "weight", 
                "acceleration", 
                "model_year", 
                "origin", 
                "car_name"] 

    @property
    def target_values(self):
        return sorted(list(set(self.filedata["mpg"])))

    @property
    def targets(self):
        return self.filedata["mpg"].astype('category').cat.codes


def uciCarEvaluationExperiment():
    """ Part 1: UCI Car Evaluation Experiment """
    carEvaluationDataObject = uciCarEvaluation()
    data = carEvaluationDataObject.data
    targets = carEvaluationDataObject.targets
    target_names = carEvaluationDataObject.target_names

    # Build the list of accuracies
    accuracyList = []

    k = 3
    classifier = KNeighborsClassifier(n_neighbors=k)

    kf = KFold(n_splits=10)
    for train_index, test_index in kf.split(data, targets):

        # Build the data/target lists
        training_data = [data.iloc[index] for index in train_index]
        training_targets = [targets.iloc[index] for index in train_index]
        testing_data = [data.iloc[index] for index in test_index]
        testing_targets = [targets.iloc[index] for index in test_index]

        # Build the model
        model = classifier.fit(training_data, training_targets)

        # Predict
        predicted_classes = model.predict(testing_data)

        accuracyList.append(accuracy_score(testing_targets, predicted_classes))

    print "The KNN model predicted cars' classes with an average accuracy of", sum(accuracyList)/len(accuracyList)


def autismDataExperiment():
    """ Part 2: Experiment on Data Related to Autism """
    autismDataObject = autismData()
    data = autismDataObject.data
    targets = autismDataObject.targets
    target_names = autismDataObject.target_names

    # Build the list of accuracies
    accuracyList = []

    k = 3
    classifier = KNeighborsClassifier(n_neighbors=k)

    kf = KFold(n_splits=10)
    for train_index, test_index in kf.split(data, targets):
        # Build the data/target lists
        training_data = [data.iloc[index] for index in train_index]
        training_targets = [targets.iloc[index] for index in train_index]
        testing_data = [data.iloc[index] for index in test_index]
        testing_targets = [targets.iloc[index] for index in test_index]

        # Build the model
        model = classifier.fit(training_data, training_targets)

        # Predict
        predicted_classes = model.predict(testing_data)

        accuracyList.append(accuracy_score(testing_targets, predicted_classes))

    print "The KNN model predicted autism's onset with an average accuracy of", sum(accuracyList)/len(accuracyList)


def automobileMPGExperiment():
    """ Part 3: Automobile MPG Experiment """
    automobileMPGDataObject = automobileMPG()
    data = automobileMPGDataObject.data
    targets = automobileMPGDataObject.targets
    target_values = automobileMPGDataObject.target_values

    # Build the list of accuracies
    meanSquareErrorList = []

    k = 3
    classifier = KNeighborsClassifier(n_neighbors=k)

    kf = KFold(n_splits=10)
    for train_index, test_index in kf.split(data, targets):

        # Build the data/target lists
        training_data = [data.iloc[index] for index in train_index]
        training_targets = [targets.iloc[index] for index in train_index]
        testing_data = [data.iloc[index] for index in test_index]
        testing_targets = [targets.iloc[index] for index in test_index]

        # Build the model
        model = classifier.fit(training_data, training_targets)

        # Predict
        predicted_classes = model.predict(testing_data)

        meanSquareErrorList.append(mean_squared_error(testing_targets, predicted_classes))

    print "The KNN model predicted cars' fuel economy with an average mean squared error of", sum(meanSquareErrorList)/len(meanSquareErrorList)

# Minimum Standard Requirements
#     Read data from text files
#     Appropriately handle non-numeric data.
#     Appropriately handle missing data.
#     Use of k-Fold Cross Validation
#     Basic experimentation on the provided datasets.

def main():
    uciCarEvaluationExperiment()
    autismDataExperiment()
    automobileMPGExperiment()

if __name__ == "__main__":
    main()
