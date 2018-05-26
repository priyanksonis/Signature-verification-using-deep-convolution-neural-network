#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 13:22:00 2018

@author: priyank
"""
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
from scipy.io import loadmat
from plots import plotresult
import numpy as np
import os


a=1
overall_FRR=0
overall_FAR=0

def SVM_WD_Train_Test(X_train,y_train,X_test_FRR,y_test_FRR,X_test_FAR,y_test_FAR):
    x,y = shuffle(X_train,y_train, random_state=2)
#    clf = svm.SVC(C=1,kernel='rbf',gamma=pow(2,-12) )
#    clf.fit(x,y)  
#    FRR=((sum(y_test_FRR)-sum(clf.predict(X_test_FRR)))/(10))*100
#    print('FRR=',FRR)
#    FAR=(sum(clf.predict(X_test_FAR))/(30))*100
#    print('FAR=',FAR)
#    #overall_FAR=overall_FAR+FAR
#    
#    ##combining testset
#    ##X_test=np.concatenate((X_test_FRR,X_test_FAR),axis=0)
#    ##y_test=np.concatenate((y_test_FRR,y_test_FAR),axis=0)
#    
    plotresult(x,y)
    
    return #FRR,FAR
    
img_data_list=[]
img_list=os.listdir('/home/priyank/Desktop/project_Luizgh/sigver_wiwd-master/gpds_signet/exploitation/')
len(img_list)
#i=0
#j=0
x=0
for img in img_list:
   x=x+1    
   print(img)
   if "forg" not in img: 
       #print(img)
       input_img=loadmat('/home/priyank/Desktop/project_Luizgh/sigver_wiwd-master/gpds_signet/exploitation'+'/'+img)['features'] #input_img.shape
       np.random.shuffle(input_img)
       X_train_ex=input_img[:14,:]  #X_train_ex.shape
       #for i in range(531):
       #  a=np.concatenate((a,a),axis=0)     
       
       X_test_FRR=input_img[14:,:] #X_test_FRR.shape
       y_test_FRR=(np.matlib.ones(10)).T  #y_test_FRR.shape
       #print(img)
       real_no=img.split("_")[1]
       X_test_FAR=loadmat('/home/priyank/Desktop/project_Luizgh/sigver_wiwd-master/gpds_signet/exploitation/forg_'+real_no)['features']
       y_test_FAR=(np.matlib.zeros(30)).T # y_test_FAR.shape   
       img_list_dev=os.listdir('/home/priyank/Desktop/project_Luizgh/sigver_wiwd-master/gpds_signet/development/')
       len(img_list_dev)
    
       img_data_list_dev=[]
       #a=0 
       #b=0
       for img1 in img_list_dev:
             if "forg" not in img1: 
               #print(img)
                   #print('a=',a)
                   #a=a+1
                   input_img=loadmat('/home/priyank/Desktop/project_Luizgh/sigver_wiwd-master/gpds_signet/development'+'/'+img1)['features']
                   np.random.shuffle(input_img)
                   input_img=input_img[:14,:]
                   img_data_list_dev.append(input_img)#len(img_data_list_dev)
             #print('b=',b)
             #b=b+1
       
       len(img_data_list_dev) 
       X_train_dev=np.concatenate(img_data_list_dev, axis=0) #X_train_dev.shape
       
       
       ##making classeas balanced
       ratio=int(X_train_dev.shape[0]/X_train_ex.shape[0])
       A=X_train_ex
       for i in range(ratio-1):
         X_train_ex=np.concatenate((X_train_ex,A),axis=0)
       
       
       X_train=np.concatenate((X_train_ex,X_train_dev),axis=0) #X_train.shape
       y_train=(np.matlib.ones(X_train_ex.shape[0]+X_train_dev.shape[0])).T  #y_train.shape
       y_train[:X_train_ex.shape[0],:]=1
       y_train[X_train_ex.shape[0]:,:]=0
       a=a+1
       frr,far=SVM_WD_Train_Test(X_train,y_train,X_test_FRR,y_test_FRR,X_test_FAR,y_test_FAR)
       overall_FRR+=frr
       overall_FAR+=far
   if x==3:
     break     
print('Avg FRR=',(overall_FRR)/a)
print('Avg FAR=',(overall_FAR)/a)



       
