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
import numpy as np
import os
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import random


a=1
overall_AUC=0
overall_FRR=0
overall_FAR=0

def SVM_WD_Train_Test(X_train,y_train,X_test_FRR,y_test_FRR,X_test_FAR,y_test_FAR):
    x,y = shuffle(X_train,y_train, random_state=2)
    clf = svm.SVC(C=1,kernel='rbf',gamma=pow(2,-12),probability=True )
    clf.fit(x,y)  
    FRR=(  (sum(y_test_FRR)-sum(clf.predict(X_test_FRR))  ) /(10))*100
    
    #class1=clf.predict(X_test_FRR)
    #print(class1)
    #prob=clf.predict_proba(X_test_FRR)
    #print(prob)
    print('FRR=',FRR)
    
    ##as1=np.array([[0]])
    FAR=(  (sum(clf.predict(X_test_FAR))-sum(y_test_FAR))  /(30))*100
    #class2=clf.predict(X_test_FAR)
    #print(class2)
    
    #prob=clf.predict_proba(X_test_FAR)
    #print(prob)
    
    
    print('FAR=',FAR)
    actual_test_X=np.concatenate((X_test_FRR,X_test_FAR),axis=0)
    actual_test_y=np.concatenate((y_test_FRR,y_test_FAR),axis=0)
    #overall_FAR=overall_FAR+FAR
    pred_test_y=clf.predict_proba(actual_test_X)
    pred_test_y=pred_test_y[:,1]
    
    false_positive_rate, true_positive_rate, thresholds = roc_curve(actual_test_y, pred_test_y)
    roc_auc = auc(false_positive_rate, true_positive_rate)

	
#    plt.title('Receiver Operating Characteristic')
#    plt.plot(false_positive_rate, true_positive_rate, 'b',
#             label='AUC = %0.2f'% roc_auc)
#    plt.legend(loc='lower right')
#    plt.plot([0,1],[0,1],'r--')
#    plt.xlim([-0.1,1.2])
#    plt.ylim([-0.1,1.2])
#    plt.ylabel('True Positive Rate')
#    plt.xlabel('False Positive Rate')
#    plt.show()


    
    
    
    return  roc_auc,FRR,FAR
    
img_data_list=[]
img_list=os.listdir('/home/priyank/Desktop/project_priyank/my_project/gpds_signet/exploitation/')
len(img_list)
#i=0
#j=0
x=0
for img in img_list:
   #x=x+1    
   #print(img)
   if "forg" not in img: 
       #print(img)
       input_img=loadmat('/home/priyank/Desktop/project_priyank/my_project/gpds_signet/exploitation'+'/'+img)['features'] #input_img.shape
       np.random.shuffle(input_img)
       X_train_ex=input_img[:14,:]  #X_train_ex.shape
       #for i in range(531):
       #  a=np.concatenate((a,a),axis=0)     
       
       X_test_FRR=input_img[14:,:] #X_test_FRR.shape
       y_test_FRR=(np.matlib.ones(10)).T  #y_test_FRR.shape
       #print(img)
       real_no=img.split("_")[1]
       X_test_FAR=loadmat('/home/priyank/Desktop/project_priyank/my_project/gpds_signet/exploitation/forg_'+real_no)['features']
       y_test_FAR=(np.matlib.zeros(30)).T # y_test_FAR.shape   
       img_list_dev=os.listdir('/home/priyank/Desktop/project_priyank/my_project/gpds_signet/development/')
       len(img_list_dev)
    
       img_data_list_dev=[]
       #a=0 
       #b=0
       for img1 in img_list_dev:
             if "forg" not in img1: 
               #print(img)
                   #print('a=',a)
                   #a=a+1
                   input_img=loadmat('/home/priyank/Desktop/project_priyank/my_project/gpds_signet/development'+'/'+img1)['features']
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
       roc_auc,frr,far=SVM_WD_Train_Test(X_train,y_train,X_test_FRR,y_test_FRR,X_test_FAR,y_test_FAR)
       overall_AUC+=roc_auc
       overall_FRR+=frr
       overall_FAR+=far
   #if x==3:
    # break     
print('Avg AUC=',(overall_AUC)/a)
print('Avg FRR=',(overall_FRR)/a)
print('Avg FAR=',(overall_FAR)/a)
       