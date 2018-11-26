install.packages('arules');
library(arules);
data(Groceries);

Groceries;
Summary(Groceries);
inspect(Groceries)
itemFrequencyPlot(Groceries, support = .2)
image(Groceries)                  


myRules <- apriori(data = Groceries,
                   parameter = list(support = .01,
                                    confidence = .0001,
                                    minlen = 2));
inspect(sort(myRules, by = "lift")[1:5]);


