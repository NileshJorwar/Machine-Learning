# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 00:23:56 2018

@author: niles
"""

import pickle
import csv
import datetime

nArr=[1,3,5,7]
xsNew=pickle.load(open('y_williams__knn__5042_k_8_njobs.pkl','rb'))
row=0
col=0
writeCSV=[]
mean=0.0
mean_sum=0.0
dateTimeNow=datetime.datetime.now()
outFile='knn_williams_5042_'+str(datetime.datetime.today().strftime('%Y-%m-%d'))+'.csv'
with open(outFile, 'a',newline='',encoding='utf-8') as myFile:
    
    writer = csv.writer(myFile)
    l=0
    for j in xsNew:
        writer.writerow(["n_neighbors= "+str(nArr[l])+'',"Result 1","Result 2","Result 3","Mean Result"])                
        for k in j.items():
            print(k[0])
            writeCSV.append(k[0])
            for i in k[1]:
                mean_sum=mean_sum+i
                writeCSV.append(i)
            mean=(mean_sum)/3
            writeCSV.append(mean)
            writer.writerow(writeCSV)
            writeCSV=[]
            mean=0.0
            mean_sum=0.0
        l=l+1