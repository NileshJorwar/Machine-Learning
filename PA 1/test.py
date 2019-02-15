# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 12:16:18 2018

@author: niles
"""

import pandas as pd
import pickle
import numpy as np
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split

print("Loading datasets...")
Xs = pickle.load(open('datasets_x.pkl', 'rb'))
ys = pickle.load(open('datasets_y.pkl', 'rb'))
print("Done.")

train_ac = np.zeros((10, 15))
test_ac = np.zeros((10, 15))


# A train_test split of your dataset Xi, yi
X_train=[]
X_test=[]
y_train=[]
y_test=[]
x={}
y={}
z={}
a={}

for i in range(10):
    xTrain,xTest,yTrain,yTest=train_test_split(Xs[i], ys[i], test_size=1/3 , random_state=int('050'))
    x[i],y[i],z[i],a[i]=train_test_split(Xs[i], ys[i], test_size=1/3 , random_state=int('050'))
    X_train.append(xTrain) 
    X_test.append(xTest)  
    y_train.append(yTrain) 
    y_test.append(yTest)
    xTrain=None
    xTest=None
    yTrain=None
    yTest=None

acc1=[]
acc2=[]
for i in range(10):
    count=1
    for depth in range(15):
        
        tree = DecisionTreeClassifier(criterion='gini',max_depth=count,random_state=int('042'))        
        #tree.fit(X_train[i],y_train[i])
        tree.fit(x[i],z[i])
        train_ac[i,depth]=tree.score(x[i],z[i])
        test_ac[i,depth]=tree.score(y[i],a[i])
        count+=1
        #print('Accuracy on Train Split {:.3f}'.format(tree.score(X_train[i],y_train[i])))
        #print('Accuracy on Train Split {:.3f}'.format(tree.score(x[i],z[i])))
        #print('Accuracy on Test Split {:.3f}'.format(tree.score(X_test[i],y_test[i])))
        
        #print('Accuracy on Test Split {:.3f}'.format(tree.score(y[i],a[i])))
        #train_ac.copy(acc1)
#test_ac.copy(acc2)    
# 2.a



#2.b


    #print('Accuracy on Test Split :{:.3f}'.format(tree.score(X_test[x],y_test[x])))
    