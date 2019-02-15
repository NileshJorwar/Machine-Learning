# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 15:14:16 2018

@author: niles
"""

import numpy as np
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split

x1_train=np.array([[True,True],[True,True],[True,True],[False,False],[False,False]])
y1_train=np.array([True,True,False,True,False])
clss=BernoulliNB(alpha=1.0,binarize=0.0,fit_prior=True,class_prior=None)
clss.fit(x1_train,y1_train)
jll=clss._joint_log_likelihood(x1_train)

print(jll)
print(clss.classes_)
print(jll.shape)
jll_op=0
for i in range(len(y1_train)):
    print(y1_train[i])
    if y1_train[i]==clss.classes_[0]:
        jll_op+=jll[i][0]
    else:
        jll_op+=jll[i][1]
print(jll_op)
print(np.exp(jll_op))