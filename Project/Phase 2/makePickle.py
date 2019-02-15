# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 19:41:15 2018

@author: niles
"""

import pickle
arr1_bush=[0.159391963,0.651077213]
arr2_williams=[0.20546737333333331,0.5625099533333334]
pickle.dump((arr1_bush), open('bush.pkl', 'wb'))
pickle.dump((arr2_williams), open('williams.pkl', 'wb'))

xsNew=pickle.load(open('bush.pkl','rb'))
print(xsNew)
xsNew1=pickle.load(open('williams.pkl','rb'))
print(xsNew1)
