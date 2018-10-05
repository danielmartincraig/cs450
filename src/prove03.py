from __future__ import print_function
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from math import sqrt
from itertools import izip
from collections import defaultdict
from more_itertools import peekable
iris = datasets.load_iris()
import os, csv, operator

class DataSetFromCSV(object):
    """ This class represents a data set that has been read from a CSV file """
    def __init__(self, filename):
        filename = "/../resources/" + filename
        with open(os.path.dirname(__file__) + filename, 'r+') as file:
            fileReader = csv.reader(file)            
            lines = [line for line in fileReader]

            self.data = self.getUSMeasurements(lines)
            self.target = self.getTargets(lines)
            self.target_names = self.getTargetNames(lines)

    def getUSMeasurements(self, lines):
        return [[float(line[0]), float(line[2]), float(line[4]), float(line[6])] for line in lines]

    def getTargets(self, lines):
        target_names = [line[-1] for line in lines]


        targetNameMap = {}
        for i, name in enumerate(target_names):
            targetNameMap[name] = i

        targets = [targetNameMap[target_name] for target_name in target_names]
        return targets

    def getTargetNames(self, lines):
        return list(set([line[-1] for line in lines]))


class KNN(object):
    """This classifier returns a KNNModel"""
    def __init__(self):
        pass

    def fit(self, k, training_data, training_targets):
        number_of_fields = len(training_data[0])
        
        means = [training_data[:,i].mean() for i in range(number_of_fields)]
        std_deviations = [training_data[:,i].std() for i in range(number_of_fields)]

        zscored_training_data = [[(x - means[i]) / std_deviations[i] for i, x in enumerate(row)] for row in training_data]

        return KNNModel(k, zscored_training_data, training_targets, means, std_deviations)

class KNNModel(object):
    """This model uses the KNN algorithm"""
    def __init__(self, k, training_data, training_targets, means, std_deviations):
        self.k = k
        self.training_data = training_data
        self.training_targets = training_targets
        self.means = means
        self.std_deviations = std_deviations

    def predict(self, testing_data):
        labels = []

        number_of_fields = len(testing_data[0])

        means = [testing_data[:,i].mean() for i in range(number_of_fields)]
        std_deviations = [testing_data[:,i].std() for i in range(number_of_fields)]        

        zscored_testing_data = [[(x - means[i]) / std_deviations[i] for i, x in enumerate(row)] for row in testing_data]        

        # Find the labels of the k nearest neighbors, and their ties
        for testing_data_row in list(zscored_testing_data):
           
            # Build the list of distances for the datapoint
            distances = []
            for training_data_row in self.training_data:
                distances.append([sqrt(sum([(testing_datapoint - training_datapoint) ** 2 for testing_datapoint, training_datapoint in izip(testing_data_row, training_data_row)]))])

            sortedDistancesWithLabels = peekable(sorted(izip(distances, self.training_targets), key=operator.itemgetter(0)))

            # Grab the k nearest neighbors
            nearestNeighbors = []
            for i in range(self.k):
                nearestNeighbors.append(sortedDistancesWithLabels.next())

            # Grab all ties
            while nearestNeighbors[-1][0] == list(sortedDistancesWithLabels.peek())[0]:
                nearestNeighbors.append(sortedDistancesWithLabels.next())
            
            # While tied, pick another neighbor
            labelHistogram = defaultdict(int)            

            # Build a histogram of the most common labels
            for label in (neighbor[1] for neighbor in nearestNeighbors):
                labelHistogram[label] += 1

            # If k is greater than 1, then while the top two are tied, pick another and update the histogram
            if self.k > 1:            
                while labelHistogram[0] == labelHistogram[1]:
                    nearestNeighbors.append(sortedDistancesWithLabels.next())
                    labelHistogram[nearestNeighbors[-1][-1]] += 1

            # Return the most frequently occuring label.  
            labels.append(sorted(labelHistogram, key=lambda (k): labelHistogram[k], reverse=True)[0])

        return labels

# Assignment Prove03
# Step 1
csvData = DataSetFromCSV("iris.csv")

data = csvData.data
target = csvData.target
target_names = csvData.target_names

# Step 2
data_train, data_test, targets_train, targets_test = train_test_split(data, target, test_size=0.33)

k = input("K?")

# Step 3
classifier = KNN()
model = classifier.fit(k, data_train, targets_train)

# Step 4
targets_predicted = model.predict(data_test)

accuracy = accuracy_score(targets_test, targets_predicted)
print("The custom KNN model predicted with an accuracy of", accuracy)

# Comparison
classifier = KNeighborsClassifier(n_neighbors=k)
model = classifier.fit(data_train, targets_train)
targets_predicted = model.predict(data_test)

accuracy = accuracy_score(targets_test, targets_predicted)
print("The KNN model from SK Learn predicted with an accuracy of", accuracy)