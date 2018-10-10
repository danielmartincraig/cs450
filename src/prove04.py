###############################################################################
#                              ID3 Decision Tree
#                                Daniel Craig
#
#                            Milestone 1 - 10/8/18
#
###############################################################################

from prove03 import UciCarEvaluation

class decisionTreeClassifier(object):
    def __init__(self):
        self.dTree = {}        

    def fit(self, training_data, training_targets):
        self.training_data = training_data
        self.training_targets = training_targets

def calculateEntropy(p):
    """ This function models the function Entropy(p) = -\sum_i{p_i\log_2{p_i}} """
    pass

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

if __name__ == "__main__":
    main()