import pandas as pd

with open("../resources/adult/adult.names") as names:
    pass

data = pd.read_csv("../resources/adult/adult.data", skipinitialspace = True, header = None, names = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "wage"])

print data[:5]