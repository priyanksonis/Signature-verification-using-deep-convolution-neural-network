#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 17:30:58 2018

@author: priyank
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.datasets import make_hastie_10_2
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
from scipy.io import loadmat
import numpy as np
import os
np.random.seed(1000)




""" HELPER FUNCTION: GET ERROR RATE ========================================="""
def get_error_rate(pred, Y):
    return sum(pred != Y) / float(len(Y))

""" HELPER FUNCTION: PRINT ERROR RATE ======================================="""
def print_error_rate(err):
    print ('Error rate: Training: %.4f - Test: %.4f' % err)

""" HELPER FUNCTION: GENERIC CLASSIFIER ====================================="""
def generic_clf(Y_train, X_train, Y_test, X_test, clf):
    clf.fit(X_train,Y_train)
    pred_train = clf.predict(X_train)
    pred_test = clf.predict(X_test)
    return get_error_rate(pred_train, Y_train), \
           get_error_rate(pred_test, Y_test)
    
""" ADABOOST IMPLEMENTATION ================================================="""
def adaboost_clf(Y_train, X_train, Y_test, X_test, M, clf):
    n_train, n_test = len(X_train), len(X_test)
    # Initialize weights
    w = np.ones(n_train) / n_train
    pred_train, pred_test = [np.zeros(n_train), np.zeros(n_test)]
    
    for i in range(M):
        # Fit a classifier with the specific weights
        clf.fit(X_train, Y_train, sample_weight = w)
        pred_train_i = clf.predict(X_train)
        pred_test_i = clf.predict(X_test)
        # Indicator function
        miss = [int(x) for x in (pred_train_i != Y_train)]
        # Equivalent with 1/-1 to update weights
        miss2 = [x if x==1 else -1 for x in miss]
        # Error
        err_m = np.dot(w,miss) / sum(w)
        # Alpha
        alpha_m = 0.5 * np.log( (1 - err_m) / float(err_m))
        # New weights
        w = np.multiply(w, np.exp([float(x) * alpha_m for x in miss2]))
        # Add to prediction
        pred_train = [sum(x) for x in zip(pred_train, 
                                          [x * alpha_m for x in pred_train_i])]
        pred_test = [sum(x) for x in zip(pred_test, 
                                         [x * alpha_m for x in pred_test_i])]
    
    #print('Y_test=',Y_test)   
    #print('pred_test=',pred_test)   
    pred_train, pred_test = np.sign(pred_train), np.sign(pred_test)
    # Return error rate in train and test set
    
    frr=get_error_rate(pred_test, Y_test)
    #print('frr=', frr*100)
    return get_error_rate(pred_train, Y_train), \
           frr*100

""" PLOT FUNCTION ==========================================================="""
def plot_error_rate(er_train, er_test):
    df_error = pd.DataFrame([er_train, er_test]).T
    df_error.columns = ['Training', 'Test']
    plot1 = df_error.plot(linewidth = 3, figsize = (8,6),
            color = ['lightblue', 'darkblue'], grid = True)
    plot1.set_xlabel('Number of iterations', fontsize = 12)
    plot1.set_xticklabels(range(0,450,50))
    plot1.set_ylabel('Error rate', fontsize = 12)
    plot1.set_title('Error rate vs number of iterations', fontsize = 16)
    plt.axhline(y=er_test[0], linewidth=1, color = 'red', ls = 'dashed')

""" MAIN SCRIPT ============================================================="""
if __name__ == '__main__':
    
       a=0
       overall_FRR=0
       overall_FAR=0
       
           
       img_data_list=[]
       img_list=os.listdir('/home/ee/mtech/eet162639/majorproject/project_priyank/sigver_wiwd-master/gpds_signet/exploitation/')
       len(img_list)
       x=0
       for img in img_list:
          x=x+1    
          if "forg" not in img: 
              #print(img)
              input_img=loadmat('/home/ee/mtech/eet162639/majorproject/project_priyank/sigver_wiwd-master/gpds_signet/exploitation'+'/'+img)['features'] #input_img.shape
              np.random.shuffle(input_img)
              X_train_ex=input_img[:14,:]  #X_train_ex.shape
              X_test_FRR=input_img[14:,:] #X_test_FRR.shape
              y_test_FRR=(np.matlib.ones(10)).T  #y_test_FRR.shape
              #print(img)
              real_no=img.split("_")[1]
              X_test_FAR=loadmat('/home/ee/mtech/eet162639/majorproject/project_priyank/sigver_wiwd-master/gpds_signet/exploitation/forg_'+real_no)['features']
              y_test_FAR=(np.matlib.zeros(30)).T # y_test_FAR.shape   
              img_list_dev=os.listdir('/home/ee/mtech/eet162639/majorproject/project_priyank/sigver_wiwd-master/gpds_signet/development/')
              len(img_list_dev)           
              img_data_list_dev=[]
              for img1 in img_list_dev:
                    if "forg" not in img1: 
                          input_img=loadmat('/home/ee/mtech/eet162639/majorproject/project_priyank/sigver_wiwd-master/gpds_signet/development'+'/'+img1)['features']
                          np.random.shuffle(input_img)
                          input_img=input_img[:14,:]
                          img_data_list_dev.append(input_img)#len(img_data_list_dev)
              len(img_data_list_dev) 
              X_train_dev=np.concatenate(img_data_list_dev, axis=0) #X_train_dev.shape
              X_train=np.concatenate((X_train_ex,X_train_dev),axis=0) #X_train.shape
              y_train=(np.matlib.ones(X_train_ex.shape[0]+X_train_dev.shape[0])).T  #y_train.shape
              y_train[:X_train_ex.shape[0],:]=1
              y_train[X_train_ex.shape[0]:,:]=-1
              
              
              Y_train=np.zeros(y_train.shape[0])
              for i in range(Y_train.shape[0]):
                     Y_train[i]=y_train[i,:]
              
              X_test=X_test_FRR
              
              
              Y_test=np.zeros(y_test_FRR.shape[0])
              for i in range(Y_test.shape[0]):
                     Y_test[i]=y_test_FRR[i,:]
              
              
           # Fit a simple decision tree first
              a=a+1
              clf_tree = DecisionTreeClassifier(max_depth = 1, random_state = 1)
              er_tree = generic_clf(Y_train, X_train, Y_test, X_test, clf_tree)
           
           # Fit Adaboost classifier using a decision tree as base estimator
           # Test with different number of iterations
              er_train, er_test = [er_tree[0]], [er_tree[1]]
              x_range = range(10, 20, 10)
              for i in x_range:    
                  print('i=',i)   
                  er_i = adaboost_clf(Y_train, X_train, Y_test, X_test, i, clf_tree)
                  er_train.append(er_i[0])
                  er_test.append(er_i[1])
                  overall_FRR=overall_FRR+er_i[1]
          
            
          print('x=',x)        
          if x==10:
                 break
          print('a=',a)
          
print('overall_FRR= ',overall_FRR)            
overall_FRR=overall_FRR/(a)  
print('overall_FRR= ',overall_FRR)    
          
           # Compare error rate vs number of iterations
              #plot_error_rate(er_train, er_test)
#          x_range = range(10, 20, 10)
#          for i in x_range:    
#                  print('i=',i)    
