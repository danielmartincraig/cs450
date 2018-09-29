from __future__ import print_function
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from math import sqrt
from itertools import izip
iris = datasets.load_iris()
import os, csv

# Step 5
class KNN(object):
    """This classifier returns a KNNModel"""
    def __init__(self):
        pass

    def fit(self, k, training_data, training_targets):
        number_of_fields = len(training_data[0])
        
        means = [training_data[:,i].mean() for i in range(number_of_fields)]
        std_deviations = [training_data[:,i].std() for i in range(number_of_fields)]

        zscored_training_data = [[(x - means[i]) / std_deviations[i] for i, x in enumerate(row)] for row in data]

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
        number_of_fields = len(testing_data[0])

        means = [testing_data[:,i].mean() for i in range(number_of_fields)]
        std_deviations = [testing_data[:,i].std() for i in range(number_of_fields)]        

        zscored_testing_data = [[(x - means[i]) / std_deviations[i] for i, x in enumerate(row)] for row in testing_data]        

        for testing_data_row in zscored_testing_data:
            distances = []
            for training_data_row in self.training_data:
                distances.append([sqrt(sum([(testing_datapoint - training_datapoint) ** 2 for testing_datapoint, training_datapoint in izip(testing_data_row, training_data_row)]))])
            print(distances)

# Assignment Prove02
# Step 1
data = iris.data
target = iris.target
target_names = iris.target_names

# Step 2
data_train, data_test, targets_train, targets_test = train_test_split(data, target, test_size=0.33)

# Step 3
classifier = KNN()
model = classifier.fit(3, data_train, targets_train)

# Step 4
targets_predicted = model.predict(data_test)

#accuracy = accuracy_score(targets_test, targets_predicted)
#print("The KNN model predicted with an accuracy of", accuracy)

