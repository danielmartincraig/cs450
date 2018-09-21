from __future__ import print_function
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
iris = datasets.load_iris()
import os, csv

# Assignment Prove01
# Step 1
data = iris.data
target = iris.target
target_names = iris.target_names

# Step 2
data_train, data_test, targets_train, targets_test = train_test_split(data, target, test_size=0.33)

# Step 3
classifier = GaussianNB()
model = classifier.fit(data_train, targets_train)

# Step 4
targets_predicted = model.predict(data_test)

accuracy = accuracy_score(targets_test, targets_predicted)
print("The GaussianNB model predicted with an accuracy of", accuracy)

# Step 5
class HardCodedClassifier(object):
    """This classifier returns a HardCodedModel"""
    def __init__(self):
        pass

    def fit(self, training_data, training_targets):
        return HardCodedModel(training_data, training_targets)

class HardCodedModel(object):
    """This model always makes the same predictions"""
    def __init__(self, training_data, training_targets):
        self.training_data = training_data
        self.training_targets = training_targets

    def predict(self, testing_data):
        return [0 for target in testing_data]

classifier = HardCodedClassifier()
model = classifier.fit(data_train, targets_train)
targets_predicted = model.predict(data_test)

accuracy = accuracy_score(targets_test, targets_predicted)

print("The hard-coded model predicted with an accuracy of", accuracy, "\n")


# Above and Beyond!
class DataSetFromCSV(object):
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


csvData = DataSetFromCSV("iris.csv")

data = csvData.data
target = csvData.target
target_names = csvData.target_names

data_train, data_test, targets_train, targets_test = train_test_split(data, target, test_size=0.33)

classifier = GaussianNB()
model = classifier.fit(data_train, targets_train)

targets_predicted = model.predict(data_test)

accuracy = accuracy_score(targets_test, targets_predicted)
print("The GaussianNB model predicted with an accuracy of", accuracy)
