# My first R script
# Daniel Craig 11/15/2018
library(e1071)
library(ggplot2)
library(caret)

# Load the dataset 
vowelDataSet <- read.csv('/home/daniel/Repos/cs450/resources/vowel/vowel.csv', head = TRUE)

# This makes a plot.  It isn't very helpful in any real way
ggplot(data = vowelDataSet) + geom_point(mapping = aes(x = F2, y = F3, color = Class)) + facet_wrap(~ Speaker, nrow = 3)

# There are 11 vowels, as evidenced by:
unique(vowelDataSet[,c('Class')])
# and
length(unique(vowelDataSet[,c('Class')]))

# Per the data's information page:
# The problem is to train the network as well as possible using only on data from "speakers" 0-47,
# and then to test the network on speakers 48-89, reporting the number of correct classifications 
# in the test set.

syllableList <- c('hid', 'hId', 'hEd', 'hAd', 'hYd', 'had', 'hOd', 'hod', 'hUd', 'hud', 'hed')

#trainSet <- data.frame()
#names(trainSet) <- c('Speaker', 'Sex', 'F0', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'Class')
syllableData <- vowelDataSet[vowelDataSet$Class==syllableList[1],]
trainIndex <- 1:48
trainData <- syllableData[trainIndex,]
testData <- syllableData[-trainIndex,]

# Grab the first instances of every syllable
for (syllable in syllableList[-1])
{
  syllableData <- vowelDataSet[vowelDataSet$Class==syllable,]
  # Shuffle the rows, to sample randomly
  syllableData <- syllableData[sample(nrow(syllableData)),]
  trainIndex <- 1:48
  tempTrainData <- syllableData[trainIndex,]
  tempTestData <- syllableData[-trainIndex,]

  trainData <- rbind(trainData, tempTrainData)
  testData <- rbind(testData, tempTestData)

  for (trialCost in 10^(1:10))
  {
    for (trialGamma in 10^(-6:-1))  
    {
      model  <- svm(Class~., data = trainData, kernel = "radial", gamma = trialGamma, cost = trialCost) 
      summary(model)
      prediction <- predict(model, testData)
      print(trialGamma)
      print('C = ')
      print(trialCost)
      print(confusionMatrix(prediction, testData$Class))
    }
  }
  
}

# Many combinations of C and gamma achieved > 97 % accuracy in my trial.  One such combination was gamma = .01 and C = 1e+07.


# Part 2 - Vowel Data

lettersDataSet <- read.csv('/home/daniel/Repos/cs450/resources/letters/letters.csv', head = TRUE)

ggplot(data = lettersDataSet) + geom_point(mapping = aes(x = onpix, y = high, color = letter)) 

trainIndex <- 1:(nrow(lettersDataSet) * .66)
lettersDataSet <- lettersDataSet[sample(nrow(lettersDataSet)),]
trainData <- lettersDataSet[trainIndex,]
testData <- lettersDataSet[-trainIndex,]
testLabels <- lettersDataSet[-trainIndex,'letter']

for (trialCost in 10^(1:2))
{
  for (trialGamma in 10^(-6:-1))  
  {
    model  <- svm(letter~., data = trainData, kernel = "radial", gamma = trialGamma, cost = trialCost) 
    summary(model)
    prediction <- predict(model, testData)
    print(trialGamma)
    print('C = ')
    print(trialCost)
    print(confusionMatrix(prediction, testLabels))
  }
}
  
# A few combinations of C and gamma achieved high accuracy in my trials.  One such combination was gamma = .1 and C = 100.
