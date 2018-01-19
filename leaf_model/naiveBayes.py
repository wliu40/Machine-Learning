# Example of Naive Bayes implemented from Scratch in Python
import csv
import random
import math
import time

'''load csv file as data resource'''	
def loadCsv(filename):
    lines = csv.reader(open(filename, "rb"))
    dataset = list(lines)
    return dataset
'''
clean the dataset
delete the first row and first column
'''
def clean(dataset):    
    dataset.pop(0)
    for i in range(0,len(dataset)):
        dataset[i].pop(0)
        for j in range(1,len(dataset[i])):
            dataset[i][j] = float(dataset[i][j])
    return dataset
    

'''
splite the input dataset to train dataset and test dataset by some ratio
'''
def splitDataset(dataset, splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    copy = list(dataset)
    while len(trainSet) < trainSize:
	index = random.randrange(len(copy))
	trainSet.append(copy.pop(index))
    return [trainSet, copy]
	
'''
separate the dataset by different classes, which is the first column,
return the class in a dictionary,
eg., returns {'Acer_Opalus': [[...], [...], [...],...],'Quercus_Rubra':[...]}
the key is the plant species, the values are intances of this species.
'''
def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
	vector = dataset[i]
	if (vector[0] not in separated):
	    separated[vector[0]] = []
	separated[vector[0]].append(vector[1:])
    return separated

'''return the mean  for a list of numbers'''
def mean(numbers):
    return sum(numbers)/float(len(numbers))
'''return the standard deviation for a list of numbers, return 0 for a single number'''
def stdev(numbers):
    avg = mean(numbers)
    if(len(numbers)==1):
	return 0.0
    variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
    return math.sqrt(variance)
	
'''
return the mean and stdev in dataset
eg. we have dataset have 2 attribute and 3 instances
dataset = [[1,2], 
           [10,4],
           [50,6]]
summarize(dataset) returns 
[(20.3, 26.1), 
 (4.0, 2.0)]
'''
def summarize(dataset):
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
	#del summaries[-1]
    return summaries
	

'''
return the summaries for each class in a dictionary
the key is the plant name, 
the value is a list contains[(mean1, stdev1), (mean2, stdev2),...(meanN, stdevN)]
'''
def summarizeByClass(dataset):
    separated = separateByClass(dataset)
    summaries = {}	
    for classValue, instances in separated.iteritems():	    
	summaries[classValue] = summarize(instances)
    return summaries
		
'''
for the attribute with continuous value, the mean and stdev are known,
calculate the probability by input a value in the test set
becareful, the case when stdev == 0
'''
def calculateProbability(x, mean, stdev):
    if (stdev == 0):
        if(x==mean):
            return 1
        return 0
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))	
    return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

'''
given an unknown instance with all the atrributes values
now, get the probabilities for each class
return the probabilites in a dictionary{'Acer_Opalus':.., 'Quercus_Rubra':..,....}
the inputvector containts all the values for all attributes
'''
def calculateClassProbabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.iteritems():
        probabilities[classValue] = 1
	for i in range(len(classSummaries)):
	    mean, stdev = classSummaries[i]
	    x = inputVector[i]
	    probabilities[classValue] *= calculateProbability(x, mean, stdev)
    return probabilities
	
'''
predict the class of the instance by inputting its values of attributes
choose the class value by its max probability
'''	
def predict(summaries, inputVector):
    probabilities = calculateClassProbabilities(summaries, inputVector)	
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.iteritems():
	if bestLabel is None or probability > bestProb:
	    bestProb = probability
	    bestLabel = classValue
    return bestLabel

'''
for each instance of the testSet, get the class prediction
'''
def getPredictions(summaries, testSet):
	predictions = []
	for i in range(len(testSet)):
	    result = predict(summaries, testSet[i][1:])	    
	    predictions.append(result)
	return predictions

'''
compare the result of prediction with the true class, return the prediction accuracy
'''
def getAccuracy(testSet, predictions):
    correct = 0
    for i in range(len(testSet)):
	if testSet[i][0] == predictions[i]:
	    correct += 1
    return (correct/float(len(testSet))) * 100.0
	

def main():    
    filename = 'train.csv'      
    splitRatio = 0.67
    accuracy_list = []
    time_list = []
    dataset = clean(loadCsv(filename))
    for k in range(10):
        
        trainingSet, testSet = splitDataset(dataset, splitRatio)
        start = time.time()   	

        summaries = summarizeByClass(trainingSet)
        predictions = getPredictions(summaries, testSet)
        end = time.time()
        time_used = end - start
        accuracy = getAccuracy(testSet, predictions)
        accuracy_list.append(accuracy)
        time_list.append(time_used)

    return [time_list, accuracy_list]    
main()
