###############################################################################
#                              ID3 Decision Tree
#                                Daniel Craig
#
#                            Milestone 1 - 10/8/18
#
###############################################################################

from prove03 import UciCarEvaluation
from numpy import log2


class decisionTreeClassifier(object):
    def __init__(self):
        self.dTree = {}        

    def fit(self, training_data, training_labels):
        self.training_data = training_data
        self.training_labels = training_labels

def recursiveTreeBuilder(data, classes):
    pass


    # Entropy(p) = -\sum_i{p_i\log_2{p_i}}
    
def calculate_entropy(labels):
    """ This function calculates the entropy of a list of labels """

    # Get a list of the unique labels that are in the list
    unique_labels = list(set(labels))

    # Declare a lambda that calculates the probability of a label occurring
    calculate_label_probability = lambda (label): labels.count(label) / float(len(labels))
 
    # Declare a generator that provides the probability of each label in the list occuring
    p = (calculate_label_probability(label) for label in unique_labels)

    # Declare a lambda that calculates the addend that represents the entropy of a single 
    # unique label in the list
    calculate_entopy_addend = lambda (p_i): -p_i*log2(p_i) if not p_i == 0 else 0
    
    # Calculate the entropy of the list by summing its parts 
    entropy = sum([calculate_entopy_addend(p_i) for p_i in p])
    
    # Return the entropy
    return entropy



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

    print calculate_entropy(['A', 'A', 'B', 'D'])

if __name__ == "__main__":
    main()