# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 15:14:16 2018

@author: niles
"""
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import train_test_split

x1_train=np.array([[2,1],[-2,1]])
y1_train=np.array([True,False])
clss=LogisticRegression(penalty='l2',C=1.0,random_state=42)
clss.fit(x1_train,y1_train)
print(x1_train)
#w0=clss.intercept_
w0=1
weights=[2,-2]
#weights=clss.coef_

jll_op=0
out_F=0
out_T=0
for i in range(len(y1_train)):
    print(y1_train[i])
    #False cLass
    if y1_train[i]==clss.classes_[0]:
        
        for j in range(len(x1_train)):
            out_F+=(weights[j]*x1_train[i][j])
        s_f=w0+out_F
    else:
        for j in range(len(x1_train)):
            out_T+=(weights[j]*x1_train[i][j])
        s_T=w0+out_T
print(jll_op)
print(np.exp(jll_op))