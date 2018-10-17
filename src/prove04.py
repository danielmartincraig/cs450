###############################################################################
#                              ID3 Decision Tree
#
#                                Daniel Craig
###############################################################################
from prove03 import UciCarEvaluation
from operator import itemgetter
from numpy import log2
from pandas import DataFrame, concat
from itertools import izip
from copy import deepcopy

class decisionTreeClassifier(object):
    def __init__(self):
        pass

    def fit(self, training_data, training_labels, feature_names):
        # Store the data
        self.training_data = training_data
        self.training_labels = training_labels
        self.feature_names = feature_names

        # Combine the data
        S = concat([self.training_data, self.training_labels], axis=1)
        S = S.rename(columns={'activity': 'labels'})

        # Build the tree
        dTree = self.recursiveTreeBuilder(S, {})

        # Return the model
        return decisionTreeModel(dTree)

    def recursiveTreeBuilder(self, S, dTree):
        # The algorithm, from the book
        
        # If all examples have the same label:
        if len(S.labels.unique()) == 1:
            return S.labels.unique()[0]
            # return a leaf with that label
        # Else if there are no features left to test:
        if len(S) == 1:
            return 'super-party'
            # return a leaf with the most common label
        # Else
        else:
            # choose the feature F^ that maximises the information gain of S to be the next node using Equation 12.2
            feature_names = list(S)[:-1]
            gains_list = [(feature_name, calculate_information_gain(S, feature_name)) for feature_name in feature_names]
            F = max(gains_list, key=itemgetter(1))[0]
            
            # add a branch from the node for each possible value f in F^
            unique_values_in_F = S[F].unique()
    
            # # for each branch:
            # for f in unique_values_in_F:
            #     # calculate S_f by removing F^ from the set of features            
            #     # recursively call the algorithm with S_f, to compute the gain relative to the current set of examples
            dTree[F] = {f: self.recursiveTreeBuilder(S.query(F+' == @f').drop(F, axis=1), {}) for f in unique_values_in_F}



        return dTree

class decisionTreeModel(object):
    def __init__(self, dTree):
        self.dTree = dTree

    def predict():
        pass


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
    classifier = decisionTreeClassifier()

    data = DataFrame({
        "deadline": ["urgent","urgent","near","none","none","none","near","near","near","urgent"],
        "party": ["yes","no","yes","yes","no","yes","no","no","yes","no"],
        "lazy": ["yes","yes","yes","no","yes","no","no","yes","yes","no"]
    })

    labels = DataFrame({
        "activity": ["party","study","party","party","pub","party","study","tv","party","study"]
    })

    feature_names = ["deadline", "party", "lazy"]

    model = classifier.fit(data, labels, feature_names)


if __name__ == "__main__":
    main()