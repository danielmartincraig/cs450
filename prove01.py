from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
iris = datasets.load_iris()

# Step 1
data = iris.data
target = iris.target
target_names = iris.target_names

# Step 2
data_train, data_test, targets_train, targets_test = train_test_split(data, target, test_size=0.33)

# Step 3
classifer = GaussianNB()
model = classifer.fit(data_train, targets_train)

# Step 4
targets_predicted = model.predict(data_test)

accuracy = accuracy_score(targets_test, targets_predicted)
print "The model predicted with", accuracy, "percent accuracy"

# Step 5
