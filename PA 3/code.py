# Programming assignment 3
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 

# Load Datasets

print("Loading datasets...")
Xs = pickle.load(open('binarized_xs.pkl', 'rb'))
ys = pickle.load(open('binarized_ys.pkl', 'rb'))
print("Done.")

# Joint Log Likelihood  on the train set. Rows are datasets, columns are decision trees with depths ranging from 1 to 15.
# Note that the arrays start with a zero index; not 1. So, the ith column should have the result of the tree that has depth (i+1).
l2_train_cll = np.zeros((10, 15))

# Joint Log Likelihood on the test set.
l2_test_cll = np.zeros((10, 15))

#Weights and Complexity
l2_num_zero_weights=np.zeros((10,15))
l1_num_zero_weights=np.zeros((10,15))
l2_model_complexity=np.zeros((10,15))

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

#L2 Regularization
for index in range(rangeValue):
    count=-7
    for depth in range(untilDepth):
        logRegression = LogisticRegression(penalty='l2',C=10**count,random_state=42)
        logRegression.fit(x_train[index],y_train[index])
        
        #L2 Model Complexity
        w0=logRegression.intercept_
        weights=logRegression.coef_
        l2_model_complexity_C=(w0**2)+np.dot(np.squeeze(weights),np.squeeze(weights)).sum()
        l2_model_complexity[index,depth]=l2_model_complexity_C
        
        ##Another way to compute complexity
# =============================================================================
#         sum1=0
#         for i in range(len(np.squeeze(weights))):
#             sum1=sum1+(np.squeeze(weights)[i]**2)
#         wnew=(w0**2)+sum1
#         
# =============================================================================
        #L2 Zero Weights
        l2_zero_weights=logRegression.coef_
        count_zero_w0=np.count_nonzero(w0==0)
        count_zero_l2=np.count_nonzero(l2_zero_weights==0)
        l2_num_zero_weights[index,depth]=(count_zero_l2+count_zero_w0)
        
        #TrainSet CLL Computation
        train_condLikelihood=logRegression.predict_log_proba(x_train[index])
        train_cl=0
        for i in range(len(y_train[index])):
            if y_train[index][i]== logRegression.classes_[0]:
                train_cl=train_cl+train_condLikelihood[i][0]
            else:
                train_cl=train_cl+train_condLikelihood[i][1]
               
        l2_train_cll[index,depth]=train_cl
        
        #TestSet CLL Computation
        test_condLikelihood=logRegression.predict_log_proba(x_test[index])
        test_cl=0
        for i in range(len(y_test[index])):
            if y_test[index][i]== logRegression.classes_[0]:
                test_cl=test_cl+test_condLikelihood[i][0]
            else:
                test_cl=test_cl+test_condLikelihood[i][1]
                
        
        l2_test_cll[index,depth]=test_cl
        count=count+1


#L1 Regularization
for index in range(rangeValue):
    count=-7
    for depth in range(untilDepth):
        logRegressionL1 = LogisticRegression(penalty='l1',C=10**count,random_state=42)
        logRegressionL1.fit(x_train[index],y_train[index])
        
        #L1 Zero Weights
        #count_zero_l1=0
        w0=logRegressionL1.intercept_
        l1_zero_weights=logRegressionL1.coef_
# =============================================================================
#         c_=0
#         for i in l1_zero_weights[0]:
#             if i == 0:
#                 c_=c_+1
#         
#         count_zero_l1=c_        
#         count_zero_l1=np.count_nonzero(l1_zero_weights==0)
# =============================================================================
        count_zero_wo=np.count_nonzero(w0==0)
        count_zero_l1=np.count_nonzero(l1_zero_weights==0)
        l1_num_zero_weights[index,depth]=(count_zero_wo+count_zero_l1)
        count=count+1
## DO NOT MODIFY BELOW THIS LINE.

for i in range(len(l2_train_cll)):
    title='Dataset '+str(i+1)
    figure_name='Dataset_Complexity_Vs_Overfit_'+str(i+1)+'.png'
    plt.plot(l2_model_complexity[i],l2_train_cll[i], marker = "*",label="Train CLL vs Model Complexity") 
    plt.plot(l2_model_complexity[i],l2_test_cll[i], marker = "*",label="Test CLL vs Model Complexity") 
    plt.ylabel('l2_train_cll/l2_test_cll')
    plt.xlabel('model complexity')
    plt.title(title)
    plt.legend(loc='center right')
    fig1 = plt.gcf()
    plt.show()
    plt.draw()
    fig1.savefig(figure_name, dpi=100)

exp_c=[]
for i in range(-7,8):
    exp_c.append(np.exp(10**i))
    
for i in range(len(l2_num_zero_weights)):
    title='Dataset '+str(i+1)
    figure_name='Dataset_Exp_Vs_Num_Zero_Weights_'+str(i+1)+'.png'
    plt.plot(exp_c,l2_num_zero_weights[i], marker = "*",label="L2_num_zero_weights vs Exponent of C") 
    plt.plot(exp_c,l1_num_zero_weights[i], marker = "*",label="L1_num_zero_weights vs Exponent of C") 
    plt.ylabel('l2_num_zero/l1_num_zero ')
    plt.xlabel('Exponent of C')
    plt.title(title)
    plt.legend(loc='center right')
    fig2 = plt.gcf()
    plt.show()
    plt.draw()
    fig2.savefig(figure_name, dpi=100)
    
print("Train set CLL")
for i in range(10):
	print("\t".join("{0:.4f}".format(n) for n in l2_train_cll[i]))
	

print("\nTest set CLL")
for i in range(10):
	print("\t".join("{0:.4f}".format(n) for n in l2_test_cll[i]))


# Once you run the code, it will generate a 'results.pkl' file. Do not modify the following code.
pickle.dump((l2_model_complexity,l2_train_cll, l2_test_cll,l2_num_zero_weights,l1_num_zero_weights), open('results.pkl', 'wb'))