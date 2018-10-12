###############################################################################
#                              ID3 Decision Tree
#                                Daniel Craig
#
#                            Milestone 1 - 10/8/18
#
###############################################################################

from prove03 import UciCarEvaluation
from operator import itemgetter
from numpy import log2

class decisionTreeClassifier(object):
    def __init__(self):
        self.dTree = {}        

    def fit(self, training_data, training_labels, feature_names):
        # Store the data
        self.training_data = training_data
        self.training_labels = training_labels
        self.feature_names = feature_names

        # Build the tree
        self.dTree = self.recursiveTreeBuilder(self.dTree, self.training_data, self.training_labels, self.feature_names)

        # Return the model
        return decisionTreeModel(self.dTree)

    def recursiveTreeBuilder(self, tree, data, labels, feature_names):
        # The algorithm, from the book

        if list(labels).count(labels[0]) == len(labels):
            # If all examples have the same label, return a leaf with that label.
        
            return [labels[0]]
        
        elif len(feature_names) == 0:
            # If there are no features left to test, return a leaf with the most common label.
            
            # TODO: decide how to determine the most common label
            # return [most_common_label]
            pass

        else:
            # choose the feature F^ that maximises the information gain of S to be the next node using Equation 12.2
            gains_list = [(feature_name, calculate_information_gain(data, feature_name)) for feature_name in feature_names]
            max_feature = max(gains_list, key=itemgetter(0))[0]
            
            # add a branch from the node for each possible value f in F^.
            values_of_max_feature = data[max_feature].unique()
            tree[max_feature] = values_of_max_feature

            # for each branch:
            for branch in tree[max_feature]:
                # calculate S_f by removing F^ from the set of features
                feature_names.remove(max_feature)
                # recursively call the algorithm with S_f, to compute the gain relative to the current set of examples
                self.recursiveTreeBuilder(tree, data[feature_names], labels, feature_names)
                pass


        return tree

class decisionTreeModel(object):
    def __init__(self, dTree):
        self.dTree = dTree

    def predict():
        pass


def calculate_information_gain(S, F):
    """ This function calculates the information gain of a particular feature """

    information_gain = 0

    # Gain(S,F)=Entropy(S) - \sum_{f\in values(F)}{\frac{\mid S_f \mid}{\mid S \mid}Entropy(S_f)}

    # Calculate the entropy of S
    # Get the subset of labels that display feature f from set of features F
    # Do the summation, and subtract the result from the entropy of S
    # Return the information gain

    return information_gain

def calculate_entropy(labels):
    """ This function calculates the entropy of a list of labels """

    # Entropy(p) = -\sum_i{p_i\log_2{p_i}}

    # Get a list of the unique labels that are in the list
    unique_labels = list(set(labels))

    # Declare a lambda that calculates the probability of a label occurring
    calculate_label_probability = lambda (label): labels.count(label) / float(len(labels))
 
    # Declare a generator that provides the probability of each label in the list occuring
    p = (calculate_label_probability(label) for label in unique_labels)

    # Declare a lambda that calculates the addend that represents the entropy of a single 
    # unique label in the list
    calculate_entropy_addend = lambda (p_i): -p_i*log2(p_i) if not p_i == 0 else 0
    
    # Calculate the entropy of the list by summing its parts 
    entropy = sum([calculate_entropy_addend(p_i) for p_i in p])
    
    # Return the entropy
    return entropy

# MILESTONE ASSIGNMENT REQUIREMENTS
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
def main():
    classifier = decisionTreeClassifier()

    uciCarEvaluationDataObject = UciCarEvaluation()

    data = uciCarEvaluationDataObject.data
    classes = uciCarEvaluationDataObject.targets
    class_names = uciCarEvaluationDataObject.target_names
    feature_names = uciCarEvaluationDataObject.attribute_names[:-1]

    model = classifier.fit(data, classes, feature_names)


if __name__ == "__main__":
    main()