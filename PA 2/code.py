# Programming assignment 2
import pickle
import numpy as np
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 

# Load Datasets

print("Loading datasets...")
Xs = pickle.load(open('binarized_xs.pkl', 'rb'))
ys = pickle.load(open('binarized_ys.pkl', 'rb'))
print("Done.")

# Joint Log Likelihood  on the train set. Rows are datasets, columns are decision trees with depths ranging from 1 to 15.
# Note that the arrays start with a zero index; not 1. So, the ith column should have the result of the tree that has depth (i+1).
train_jll = np.zeros((10, 15))

# Joint Log Likelihood on the test set.
test_jll = np.zeros((10, 15))



############ TODO ############

# Write your code below this line.

x_train={}
x_test={}
y_train={}
y_test={}

rangeValue=len(Xs)
untilDepth=15
for index in range(rangeValue):
    x_train[index],x_test[index],y_train[index],y_test[index]=train_test_split(Xs[index], ys[index], 
           test_size=1./3 , random_state=int('5042'))

for index in range(rangeValue):
    count=-7
    for depth in range(untilDepth):
        naiveBayes = BernoulliNB(alpha=10**count,binarize=0.0,class_prior=None,
                                      fit_prior=True)        
        naiveBayes.fit(x_train[index],y_train[index])
        
        #TrainSet JLL Computation
        train_jointLikelihood=naiveBayes._joint_log_likelihood(x_train[index])
        train_jl=0
        for i in range(len(y_train[index])):
            if y_train[index][i]== naiveBayes.classes_[0]:
                train_jl=train_jl+train_jointLikelihood[i][0]
            else:
                train_jl=train_jl+train_jointLikelihood[i][1]
               
        train_jll[index,depth]=train_jl
        
        #TestSet JLL Computation
        test_jointLikelihood=naiveBayes._joint_log_likelihood(x_test[index])
        test_jl=0
        for i in range(len(y_test[index])):
            if y_test[index][i]== naiveBayes.classes_[0]:
                test_jl=test_jl+test_jointLikelihood[i][0]
            else:
                test_jl=test_jl+test_jointLikelihood[i][1]
                
        
        test_jll[index,depth]=test_jl
        count+=1

## DO NOT MODIFY BELOW THIS LINE.

#plt.plot(train_jll[2], color = 'blue', marker = "*") 
#plt.plot(test_jll[2], color = 'red', marker = "*") 
#plt.show()

print("Train set JLL")
for i in range(10):
	print("\t".join("{0:.4f}".format(n) for n in train_jll[i]))
	

print("\nTest set JLL")
for i in range(10):
	print("\t".join("{0:.4f}".format(n) for n in test_jll[i]))


# Once you run the code, it will generate a 'results.pkl' file. Do not modify the following code.
pickle.dump((train_jll, test_jll), open('results.pkl', 'wb'))