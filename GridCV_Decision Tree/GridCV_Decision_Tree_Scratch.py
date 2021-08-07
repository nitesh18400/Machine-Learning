import h5py
import pandas as pd
import numpy as np
import statistics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import pickle
from sklearn.decomposition import PCA
#-------------------------------------------------------------------------------------
#Grid Search For optimal depth finding
def grid_search_depthdt_kfold(decison_tree_object,dfx,dfy,param_grid,k):
    depth=param_grid["depth"]
    acc_with_depth={}
    train_acc_with_depth={}
    for d in range(len(depth)):
        fold_size = len(dfx)//k
        avg=[]
        train_avg=[]
        for i in range(k):
            split_x = np.vsplit(dfx, np.array([i*fold_size, (i+1)*fold_size]))
            split_y = np.split(dfy, [i*fold_size, (i+1)*fold_size])
            test_split_x = split_x[1]
            train_split_x = np.vstack((split_x[0], split_x[2]))
            test_split_y = split_y[1]
            train_split_y = np.concatenate((split_y[0], split_y[2]))
            decison_tree_object.max_depth=depth[d]
            decison_tree_object.fit(train_split_x, train_split_y)
            train_avg.append(decison_tree_object.score(train_split_x, train_split_y))
            avg.append(decison_tree_object.score(test_split_x,test_split_y))
        train_acc_with_depth[depth[d]]=statistics.mean(train_avg)
        acc_with_depth[depth[d]]=statistics.mean(avg)
    v = list(acc_with_depth.values())
    k = list(acc_with_depth.keys())
    return k[v.index(max(v))],max(v),acc_with_depth,train_acc_with_depth   
#------------------------------------------------------------------------------------------
#Kfold for Decision Tree and GNB 
def k_fold_DT_GNB(dfx,dfy,optimal_depth_for_DT,k):
    d=[]
    g=[]
    fold_size=len(dfx)//k
    for i in range(k):
        split_x = np.vsplit(dfx, np.array([i*fold_size,(i+1)*fold_size]))
        split_y = np.split(dfy,[ i*fold_size, (i+1)*fold_size])
        test_split_x = split_x[1]
        train_split_x = np.vstack((split_x[0], split_x[2]))
        test_split_y = split_y[1]
        train_split_y = np.concatenate((split_y[0], split_y[2]))
        dtc = DecisionTreeClassifier(max_depth=optimal_depth_for_DT,random_state=1)
        dtc.fit(train_split_x, train_split_y)
        dts=dtc.score(test_split_x,test_split_y)
        d.append(dts)
        gnb = GaussianNB()
        gnb.fit(train_split_x,train_split_y)
        sc=gnb.score(test_split_x, test_split_y)
        g.append(sc)
    #saving model with accuracy
    if(statistics.mean(d)>statistics.mean(g)):
        filename = 'best_model.sav'
        pickle.dump(dtc, open(filename, 'wb'))
    else:
        filename = 'best_model.sav'
        pickle.dump(gnb, open(filename, 'wb'))
        
    return statistics.mean(d),statistics.mean(g)