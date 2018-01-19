import numpy as np
import csv
import random
from sklearn import tree
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


def main():
    filename = 'train.csv'
    my_data = clean(loadCsv(filename))
    print 'sklearn tree'

    splitRatio = 0.67
    
    accuracy_list = []
    time_list = []
    
    start = time.time()
    for k in range(10):
        trainingSet, testSet = splitDataset(my_data, splitRatio)
        Y = np.array([instance[0] for instance in trainingSet])
        X = np.array([instance[1:] for instance in trainingSet])
        clf1 = tree.DecisionTreeClassifier()

        clf1.fit(X, Y)   
        correct = 0
        for i in range(len(testSet)):
            result = clf1.predict([testSet[i][1:]])        
            if result == testSet[i][0]:
                correct += 1
        accuracy = (correct/float(len(testSet))) * 100.0
        end = time.time()
        time_used = end - start
        accuracy_list.append(accuracy)
        time_list.append(time_used)
       
    return [time_list, accuracy_list]   

main()