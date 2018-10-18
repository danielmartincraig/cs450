###############################################################################
#                              ID3 Decision Tree
#
#                                Daniel Craig
###############################################################################
from operator import itemgetter
from numpy import log2
from sklearn import tree
from pandas import DataFrame, concat, read_csv
from itertools import izip
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import pydot


class datasetFromCsv(object):
    """ This class represents a dataset; the data is read from a csv file """
    def __init__(self, sep):
        self.filedata = read_csv(self.filename, header = None, names = self.attribute_names, skipinitialspace=True, sep=sep)


class UciCarEvaluation(datasetFromCsv):
    """ This class represents the UCI data """
    def __init__(self):
        super(UciCarEvaluation, self).__init__(sep=',')

    @property
    def data(self):
        return DataFrame({
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
    def attribute_names(self):
        return ["buying", 
                "maint", 
                "doors", 
                "persons", 
                "lug_boot", 
                "safety", 
                "class"]

    @property
    def target_names(self):
        return sorted(list(set(self.filedata[self.attribute_names[-1]])))

    @property
    def targets(self):
        return DataFrame({'labels': self.filedata[self.attribute_names[-1]].astype('category').cat.codes})


class decisionTreeClassifier(object):
    def __init__(self):
        pass

    def fit(self, training_data, training_labels):
        # Store the data
        self.training_data = training_data
        self.training_labels = training_labels

        # Combine the data
        S = concat([self.training_data, self.training_labels], axis=1)
        S = S.rename(columns={'activity': 'labels'})

        # Build the tree
        dTree = self.recursiveTreeBuilder(S)

        # Return the model
        return decisionTreeModel(dTree)

    def recursiveTreeBuilder(self, S):
        # The algorithm, from the book
        dTree = {}

        # If all examples have the same label:
        if len(S.labels.unique()) == 1:
            # return a leaf with that label
            return S.labels.unique()[0]

        # Else if there are no features left to test:
        if len(S) == 1:
            # return a leaf with the most common label
            labels = S.labels
            return max(set(labels), key=labels.count)
        
        # Else
        else:
            # choose the feature F^ that maximises the information gain of S to be the next node using Equation 12.2
            feature_names = list(S)[:-1]
            gains_list = [(feature_name, calculate_information_gain(S, feature_name)) for feature_name in feature_names]
            F = max(gains_list, key=itemgetter(1))[0]
            
            # add a branch from the node for each possible value f in F^
            unique_values_in_F = S[F].unique()
    
            # # for each branch:
            #     # calculate S_f by removing F^ from the set of features            
            #     # recursively call the algorithm with S_f, to compute the gain relative to the current set of examples
            dTree[F] = {f: self.recursiveTreeBuilder(S.query(F+' == @f').drop(F, axis=1)) for f in unique_values_in_F}

        return dTree


class decisionTreeModel(object):
    def __init__(self, dTree):
        self.dTree = dTree

    def _predicting_visit(self, dTree, data_vector):
        # Get the attribute that the decision will be based on
        attribute = dTree.keys()[0]
        attributeValue = getattr(data_vector, attribute)
        subtree = dTree[attribute][attributeValue]

        try:
            if isinstance(subtree, dict):
                subtree = self._predicting_visit(subtree, data_vector)
        except KeyError:
            # Unseen data, get the mode of the remaining labels in the subtree
            remaining_labels = get_remaining_labels(subtree)
            subtree = max(set(remaining_labels), key=remaining_labels.count)

        return subtree

    def predict(self, testing_data):
        return [self._predicting_visit(self.dTree, data_vector) for data_vector in testing_data.itertuples()]
            
    def _draw(self, graph, parent_name, child_name):
        edge = pydot.Edge(parent_name, child_name)
        graph.add_edge(edge)

    def _drawing_visit(self, graph, node, parent=None):
        nvl = lambda x, y: y if None == x else x  

        for k, v in node.iteritems():
            if isinstance(v, dict):
                if parent:
                    self._draw(graph, parent, nvl(parent, '') + '>' + k)
                self._drawing_visit(graph, v, nvl(parent, '')+'>'+k)
            else:
                self._draw(graph, parent, nvl(parent, '') + '>' + k)
                self._draw(graph, nvl(parent, '') + '>' + k, k+'_'+v)

    def drawTree(self):
        """ Draw the tree - this method draws very heavily from https://stackoverflow.com/questions/13688410/dictionary-object-to-decision-tree-in-pydot"""
        graph = pydot.Dot(graph_type='graph')
        self._drawing_visit(graph, self.dTree)
        graph.write_png('myGraph.png')

def get_remaining_labels(subtree):
    labels = []

    # Add them to the label list, and recurse
    if isinstance(subtree, dict):
        branches = subtree.keys()
        for branch in branches: 
            labels += get_remaining_labels(subtree[branch])
    else:
        labels += [subtree]
    
    return labels

def calculate_information_gain(S, F):
    """ This function calculates the information gain of a particular feature """

    # Gain(S,F)=Entropy(S) - \sum_{f\in values(F)}{\frac{\mid S_f \mid}{\mid S \mid}Entropy(S_f)}

    # Initialize the information gain variable
    information_gain = 0

    # Calculate the entropy of S
    entropy_of_s = calculate_entropy(S.labels)
    
    # Create a list of the values of S_f 
    # Consider it to look like the following, although they are actually pandas vectors 
    # [['party', 'study', 'study'], ['party', 'study', 'tv', 'party'], ['party', 'pub', 'party']]
    unique_values_in_F = S[F].unique()
    values_of_S_f = [S.query(F+' == @f').labels for f in unique_values_in_F]

    # Create a list of the weights - len(S_f) / len(S)
    list_of_weights = [len(values_set) / float(len(S)) for values_set in values_of_S_f]

    # Create a list of the entropies
    list_of_entropies = [calculate_entropy(values_set) for values_set in values_of_S_f]

    # Find the weighted sum
    weighted_pairs = izip(list_of_weights, list_of_entropies)
    weighted_sum_of_f_in_F = sum([weight * entropy for weight, entropy in weighted_pairs])

    # Subtract from the entropy of S
    information_gain = entropy_of_s - weighted_sum_of_f_in_F

    # Return the information gain
    return information_gain

def calculate_entropy(labels):
    """ This function calculates the entropy of a list of labels """

    # Entropy(p) = -\sum_i{p_i\log_2{p_i}}

    # Get a list of the unique labels that are in the list
    unique_labels = labels.unique()

    # Declare a lambda that calculates the probability of a label occurring
    calculate_label_probability = lambda (label): list(labels).count(label) / float(len(labels))
 
    # Declare a generator that provides the probability of each label in the list occuring
    p = (calculate_label_probability(label) for label in unique_labels)

    # Declare a lambda that calculates the addend that represents the entropy of a single 
    # unique label in the list
    calculate_entropy_addend = lambda (p_i): -p_i*log2(p_i) if not p_i == 0 else 0
    
    # Calculate the entropy of the list by summing its parts 
    entropy = sum([calculate_entropy_addend(p_i) for p_i in p])
    
    # Return the entropy
    return entropy

# ASSIGNMENT REQUIREMENTS
# Make sure you are familiar with the instructions for the complete assignment.
# 
# Create a class for your decision tree classifier.
# 
# Ensure that you can read in at least one dataset with discrete values and
#    send it to your classifier. It is just fine to pre-process the data in another
#    program or script to prepare it, but you should have at least some data for
#    your algorithm.
# 
# Have a mechanism for representing your tree, along with branches, leaf nodes,
#    and how it will store attributes and target values.
# 
# Have a function that can calculate entropy, and verify that it computes the
#    correct value (check it by hand) for cases with both two classes, and also more
#    than two classes.
# 
# Have a recursive function to build a tree that can partition the data it
#    receives according to the value of an attribute. And that can call itself.
#
# Have the logic in place for at least the base cases of the recursive algorithm.
# 
# Implement a new algorithm, the ID3 Decision Tree.
# 
# After implementing the algorithm, use it to classify a dataset of your choice.
# 
# Compare your implementation of the ID3 algorithm to an existing one (e.g.,
#    scikit-learn, Weka) and compare/contrast the results.
# 
# When complete, push your code to a public GitHub repository and answer the
# questions in the submission text file. Please fill it out and upload it to
# I-Learn.


def main():
    #########################################################
    # Test my decision tree classifier
    #########################################################

    classifier = decisionTreeClassifier()

    uciCarEvaluationDataObject = UciCarEvaluation()
    data = uciCarEvaluationDataObject.data
    labels = uciCarEvaluationDataObject.targets
    label_names = uciCarEvaluationDataObject.target_names

    model = classifier.fit(data, labels)
    
    accuracy_list = []
    kf = KFold(n_splits=10)
    for train_index, test_index in kf.split(data, labels):

        # Build the data/target lists
        training_data = data.iloc[train_index] 
        training_labels = labels.iloc[train_index]
        testing_data = data.iloc[test_index]
        testing_labels = labels.iloc[test_index]

        # Build the model
        model = classifier.fit(training_data, training_labels)

        # # Predict
        predicted_classes = model.predict(testing_data)
        
        accuracy_list.append(accuracy_score(testing_labels, predicted_classes))
    
    print "The custom decision tree predicted the auto dataset's classes with an average of",
    print sum(accuracy_list) / float(len(accuracy_list)) * 100,
    print "percent accuracy." 

    #########################################################
    # Compare the SK-learn decision tree classifier
    #########################################################

    classifier = tree.DecisionTreeClassifier()

    model = classifier.fit(data, labels)
    
    accuracy_list = []
    kf = KFold(n_splits=10)
    for train_index, test_index in kf.split(data, labels):

        # Build the data/target lists
        training_data = data.iloc[train_index] 
        training_labels = labels.iloc[train_index]
        testing_data = data.iloc[test_index]
        testing_labels = labels.iloc[test_index]

        # Build the model
        model = classifier.fit(training_data, training_labels)

        # # Predict
        predicted_classes = model.predict(testing_data)
        
        accuracy_list.append(accuracy_score(testing_labels, predicted_classes))
    print "The sk-learn decision tree predicted the auto dataset's classes with an average of",
    print sum(accuracy_list) / float(len(accuracy_list)) * 100,
    print "percent accuracy." 

if __name__ == "__main__":
    main()