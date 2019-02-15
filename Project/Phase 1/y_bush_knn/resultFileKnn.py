# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 00:23:56 2018

@author: niles
"""

import pickle
import csv
import datetime
xsArr=[]
nArr=[]
for m in range(6):
    if m%2==1:
        xs=pickle.load(open('res_stratefied_knn_'+str(m)+'_5230_k_8_njobs.pkl','rb'))
        xsArr.append(xs)
        nArr.append(m)
        xs=None
row=0
col=0
writeCSV=[]
mean=0.0
mean_sum=0.0
dateTimeNow=datetime.datetime.now()
outFile='knn_bush_5230'+str(datetime.datetime.today().strftime('%Y-%m-%d'))+'.csv'
with open(outFile, 'a',newline='',encoding='utf-8') as myFile:
    
    writer = csv.writer(myFile)
    l=0
    for j in xsArr:
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