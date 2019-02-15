# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 19:41:15 2018

@author: niles
"""

import pickle
arr1_bush=[0.139423483345142,0.0944056706115438,0.0353273853964461,0.658792842103802,0.159391963,0.651077213,0.7248322147651007,1.0]
arr2_williams=[0.126754083275822,0,0,0.543022774327122,0.20546737333333331,0.5625099533333334,0.5217391304347826,0.9859154929577464]
pickle.dump((arr1_bush), open('bush.pkl', 'wb'))
pickle.dump((arr2_williams), open('williams.pkl', 'wb'))

xsNew=pickle.load(open('bush.pkl','rb'))
print(xsNew)
xsNew1=pickle.load(open('williams.pkl','rb'))
print(xsNew1)
