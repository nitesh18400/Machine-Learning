#Importing libraries
#-------------------------------------------------------------------------------------
import h5py
import pandas as pd
import numpy as np
import math
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
#-------------------------------------------------------------------------------------
#GNB Implementation
class GassianNaiveBayes:
    m_v={}
    pr={}
    def __init__(self):
        pass
    #Data Cluster according to classes
    def grouping(self,X,Y):
        G={}
        for i in range(len(np.unique(Y))):
            G[i]=np.empty((0, X.shape[1]), dtype=np.float64)
            
        for i in range(len(Y)):
            G[Y[i]]=np.append(G[Y[i]],np.array([X[i]]),axis=0)
        return G
    #finding mean and varience
    def mean_var(self,G):
        mv={}
        pr={}
        for i in range(len(G)):
            mv[i] = np.empty((0,len(G[0][0])), dtype=np.float64)
        for i in range(len(G)):
            mn=(G[i].mean(axis=0))
            var=(G[i].var(axis=0))
            mv[i] = np.append(mv[i],np.array([mn]),axis=0)
            mv[i] = np.append(mv[i],np.array([var]),axis=0)
        return mv
    #GNB formula implementation
    def GNBprob(self,x,u,v):
        try:
            fir=1/(math.sqrt(2*3.14*v))
            sec=math.exp(-1*(((x-u)**2)/(2*v)))
        except ZeroDivisionError as e:
            return 0
        return fir*sec        
    #fit function
    def fit(self,X,Y):
        for i in np.unique(Y):
            self.pr[i]=np.count_nonzero(Y==i)
        for i in np.unique(Y):
            self.pr[i]=self.pr[i]/len(Y)        
        G=self.grouping(X,Y)
        self.m_v=self.mean_var(G);
        pass
    #formula ouput 
    def predict_helper(self,x_row,mean,var,c):
        p=math.log(self.pr[c])
        for i in range(len(x_row)):
            p+=math.log(self.GNBprob(x_row[i],mean[i],var[i]))
        return p
    #predict function
    def predict(self,X):
        predicted_y=[]
        for i in range(len(X)):
            temp=[]
            for j in range(len(self.m_v)):
                temp.append(self.predict_helper(X[i],self.m_v[j][0],self.m_v[j][1],j))
            predicted_y.append(temp.index(max(temp)))
        return predicted_y
    #predict scoring
    def predict_score(self,oy,py):
        sc=0
        for i in range(len(oy)):
            if(oy[i]==py[i]):
                sc+=1
        return (sc/len(oy))

#------------------------------------------------------------------------------------------------
print("Dataset-1")
# Dataset-1
hf = h5py.File('part_A_train.h5', mode='r')
dfx = hf.get('X')[()]
dfy = hf.get('Y')[()]
dfy = np.where(dfy == 1)[1]
#--------------------------------------------------------------------------------------------------
#spliitting
X_train, X_test, y_train, y_test = train_test_split(dfx, dfy, test_size=0.2, random_state=1)
#--------------------------------------------------------------------------------------------------
# Dimension Reduction
pca = PCA(n_components=50)
pca.fit(X_train)
X_train= pca.transform(X_train)
X_test=pca.transform(X_test)
#----------------------------------------------------------------------------------------------------
#scratch implementation
gn=GassianNaiveBayes()
gn.fit(X_train,y_train)
py=gn.predict(X_test)
print("Scratch Implementation accuracy:",gn.predict_score(y_test,py))
#----------------------------------------------------------------------------------------------------
#sklearn Implmentation
gnsk=GaussianNB()
gnsk.fit(X_train,y_train)
print("Sklearn Algorithm accuracy:",gnsk.score(X_test,y_test))
#---------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------
print("Dataset-2")
# Dataset-1
hf = h5py.File('part_B_train.h5', mode='r')
dfx = hf.get('X')[()]
dfy = hf.get('Y')[()]
dfy = np.where(dfy == 1)[1]
#--------------------------------------------------------------------------------------------------
#spliitting
X_train, X_test, y_train, y_test = train_test_split(
    dfx, dfy, test_size=0.2, random_state=1)
#--------------------------------------------------------------------------------------------------
#Dimension Reduction
pca = PCA(n_components=40)
pca.fit(X_train)
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)
#----------------------------------------------------------------------------------------------------
#scratch implementation
gn = GassianNaiveBayes()
gn.fit(X_train, y_train)
py = gn.predict(X_test)
print("Scratch Implementation accuracy:", gn.predict_score(y_test, py))
#----------------------------------------------------------------------------------------------------
#sklearn Implmentation
gnsk = GaussianNB()
gnsk.fit(X_train, y_train)
print("Sklearn Algorithm accuracy:", gnsk.score(X_test, y_test))
#---------------------------------------------------------------------------------------------------
