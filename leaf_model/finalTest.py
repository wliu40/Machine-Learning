import decisionTreeTest
import naiveBayes
import sklearn_bayes
import sklearn_decisiontree
import matplotlib.pyplot as plt
import numpy as np
#naiveBayes.main()

def avg(x):
    return float(sum(x))/len(x)

time_1, accuracy_1 = decisionTreeTest.main()
time_2, accuracy_2 = naiveBayes.main()
time_3, accuracy_3 = sklearn_bayes.main()
time_4, accuracy_4 = sklearn_decisiontree.main()


print 'decision tree average time consumed: ' + str(avg(time_1))
print 'decision tree average accuracy: ' + str(avg(accuracy_1))

print 'naive Bayes average time consumed: '+ str(avg(time_2))
print 'naive Bayes average accuracy: '+ str(avg(accuracy_2))

print 'sklearn Bayes average time consumed: '+ str(avg(time_3))
print 'sklearn Bayes average accuracy: '+ str(avg(accuracy_3))

print 'sklearn decision tree average time consumed: '+ str(avg(time_4))
print 'sklearn decision tree average accuracy: '+ str(avg(accuracy_4))




f1 = plt.figure(1)
plt.xlabel('Experiment number')
plt.ylabel('Run time')
plt.plot(time_1, 'gx', label = 'decision_tree')
plt.plot(time_2, 'rv', label = 'naive_bayes')
plt.plot(time_3, 'ko', label = 'sklearn_bayes')
plt.plot(time_4, 'bs', label = 'sklearn_tree')
plt.legend(loc = 'best')
f1.show()

f2 = plt.figure(2)
plt.xlabel('Experiment number')
plt.ylabel('Accuracy')
plt.plot(accuracy_1, 'gx', label = 'decision_tree')
plt.plot(accuracy_2, 'rv', label = 'naive_bayes')
plt.plot(accuracy_3, 'ko', label = 'sklearn_bayes')
plt.plot(accuracy_4, 'bs', label = 'sklearn_tree')
plt.legend(loc = 'best')
f2.show()


