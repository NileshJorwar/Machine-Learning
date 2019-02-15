# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 19:41:15 2018

@author: niles
"""

import pickle
arr1_bush=[0.139423483345142,0.0944056706115438,0.0353273853964461,0.658792842103802]
arr2_williams=[0.126754083275822,0,0,0.543022774327122]
pickle.dump((arr1_bush), open('bush.pkl', 'wb'))
pickle.dump((arr2_williams), open('williams.pkl', 'wb'))

xsNew=pickle.load(open('bush.pkl','rb'))
print(xsNew)
xsNew1=pickle.load(open('williams.pkl','rb'))
print(xsNew1)
