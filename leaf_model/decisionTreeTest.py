import numpy as np
import csv
import random
import time
  
'''load csv file as data resource'''	
def loadCsv(filename):    
    lines = csv.reader(open(filename, "rb"))
    dataset = list(lines)
    return dataset
	
'''
clean the dataset
delete the first row and first column,
because the first colmun is id,
the first row is header.
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
Divides a set on a specific column.Can handle numeric or nominal values
eg. divideset(my_data, 2, 0.04) will divide the my_data to two sets based on
the 2nd column's value, those >=0.04 will be in one set, otherwise in another set
'''
def divideset(rows,column,value):
    
   # Make a function that tells us if a row is in the first group (true) or the second group (false)
    split_function=None
    if isinstance(value,int) or isinstance(value,float): # check if the value is a number i.e int or float
       split_function=lambda row:row[column]>=value
    else:
       split_function=lambda row:row[column]==value
   
   # Divide the rows into two sets and return them
    set1=[row for row in rows if split_function(row)]
    set2=[row for row in rows if not split_function(row)]
    return (set1,set2)

'''
given a dataset, return a dictionary.
the key is the class name (the class name is in the first column)
the value is the number of this class in whole dataset
'''
def uniquecounts(rows):
    results={}
    for row in rows:
      # The result is the first column
       r=row[0]      
       if r not in results: results[r]=0
       results[r]+=1
    return results

'''
Entropy is the sum of p(x)log(p(x)) across all the different possible 
results, return the entropy for a dataset

'''
def entropy(rows):      
    from math import log
    log2=lambda x:log(x)/log(2)  
    results=uniquecounts(rows)
    # Now calculate the entropy
    ent=0.0
    for r in results.keys():     
        p=float(results[r])/len(rows)
        ent=ent-p*log2(p)
    return ent
   
'''
col: the column number to be tested
value: the criteria to divide the set, eg. if x> 0.04 go to right node, otherwise left
results: if it is a internal node, result is 'None', if it is a leaf node, result will contains
the class name and the number of intances
tb and fb are decisionnodes, which are the next nodes in the tree if the result is true or false, 
respectively (e.g. go to node tb or fb).
'''
class decisionnode:        
    def __init__(self,col=-1,value=None,results=None,tb=None,fb=None):
        self.col=col 
        self.value=value 
        self.results=results
        self.tb=tb 
        self.fb=fb

'''
Build the decision tree recursively.
rows is the set, either whole dataset or part of it in the recursive call, 
scoref is the method to measure heterogeneity. By default it's entropy.

'''
def buildtree(rows,scoref=entropy):                                   
    if len(rows)==0: 
        return decisionnode() #len(rows) is the number of units in a set
    current_score=scoref(rows)

  # Set up some variables to track the best criteria
    best_gain=0.0
    best_criteria=None
    best_sets=None
  
    column_count=len(rows[0])  #count the # of attributes/columns
    #start from the 2nd column, since the 1st column is the class name
    #for each column, find the distinct values in that colulmn                
    for col in range(1,column_count):
      
    # Generate the list of all possible different values in the considered column
        global column_values
        column_values={}            
        for row in rows:
            column_values[row[col]]=1
    # Now try dividing the rows up for each value in this column
        for value in column_values.keys():            
            #the 'values' here are the keys of the dictionnary
            (set1,set2)=divideset(rows,col,value) #define set1 and set2 as the 2 children set of a division
        
            # Information gain
            p=float(len(set1))/len(rows) #p is the size of a child set relative to its parent
            gain=current_score-p*scoref(set1)-(1-p)*scoref(set2) #cf. formula information gain
            if gain>best_gain and len(set1)>0 and len(set2)>0: #set must not be empty
                best_gain=gain
                best_criteria=(col,value)
                best_sets=(set1,set2)
        
  # Create the sub branches   
    if best_gain>0:        
        trueBranch=buildtree(best_sets[0])
        falseBranch=buildtree(best_sets[1])
        return decisionnode(col=best_criteria[0],value=best_criteria[1],
                        tb=trueBranch,fb=falseBranch)
    else:
        return decisionnode(results=uniquecounts(rows))
   
'''print the tree recursively'''
def printtree(tree,indent=''):    
   # Is this a leaf node?
    if tree.results!=None:
        print(str(tree.results))
    else:
        print(str(tree.col)+':'+str(tree.value)+'? ')
        # Print the branches
        
        print indent + 'T->',
        printtree(tree.tb,indent+'  ')
        print indent+'F->',
        printtree(tree.fb,indent+'  ')
        
'''classify a test case using the tree model'''     
def classify(observation,tree):        
    if tree.results!=None:
        return tree.results
    else:
        v=observation[tree.col]
        branch=None
        if isinstance(v,int) or isinstance(v,float):
            if v>=tree.value: 
                branch=tree.tb
            else: 
                branch=tree.fb
        else:
            if v==tree.value: 
                branch=tree.tb
            else: 
                branch=tree.fb
        return classify(observation,branch)
    
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
    my_data = clean(loadCsv(filename))
    splitRatio = 0.67
    print 'decision tree'
    accuracy_list = []
    time_list = []
    
    for k in range(10):        
        trainingSet, testSet = splitDataset(my_data, splitRatio)
        start = time.time()
        tree=buildtree(trainingSet)
        correct = 0
        for i in range(len(testSet)):
            result = classify(testSet[i][1:],tree)            
            if list(result.keys())[0] == testSet[i][0]:
                correct += 1
        end = time.time()       
        time_used = end - start
        accuracy = correct/float(len(testSet)) * 100.0
        
        time_list.append(time_used)
        accuracy_list.append(accuracy)

    return [time_list, accuracy_list]

    

    
    
    
    
    
