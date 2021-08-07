import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
class MyLogisticRegression():                                     # logistics regression
    features_parameter_SGD = []
    features_parameter_BGD = []
    val_x = pd.DataFrame()
    val_y = pd.DataFrame()
    test_x = pd.DataFrame()                                 # various variable used for loss plots and prediction  
    test_y = pd.DataFrame()
    alpha_SGD, ite_SGD = 0, 0
    alpha_BGD, ite_BGD = 0, 0

    def __init__(self):
        pass

    def set_ite_alpha(self, a, b, c, d):                               # set learning rate and epochs
        self.alpha_SGD, self.ite_SGD = a, b
        self.alpha_BGD, self.ite_BGD = c, d

    def hypothesis(self, features_parameter, features, row):                #hyposthesis
        sm = features_parameter[0]
        sm += np.matmul(features[row], np.array(features_parameter[1:]).T)
        temp = 1.0/(1.0+np.exp(-sm))
        return temp

    def cost_function(self, outputp, feature_parameter, inputp):               # cost function for losgistic regression
        sumission = 0
        j = 0
        for i in range(len(outputp)):
            j += ((outputp[i]*np.log(self.hypothesis(feature_parameter, inputp, i))) +
                  ((1-outputp[i])*(np.log(1-self.hypothesis(feature_parameter, inputp, i)))))

        m = len(outputp)

        return -1*(j/m)

    def sgd(self, outputp, features_parameter, inputp, alpha):                  #update theta sgd
        temp_parameter = [0]*(len(features_parameter))
        temp = random.randint(0, 7*(len(outputp)//10))
        for i in range(len(features_parameter)):
            if(i == 0):
                temp_parameter[i] += (self.hypothesis(features_parameter,
                                                      inputp, temp)-outputp[temp])
            else:
                temp_parameter[i] += (self.hypothesis(features_parameter,
                                                      inputp, temp)-outputp[temp])*inputp[temp, i-1]
        for i in range(len(features_parameter)):
            features_parameter[i] -= ((alpha*temp_parameter[i]))
        return features_parameter

    def bgd(self, outputp, features_parameter, inputp, alpha):                   # update theta bgd
        temp_parameter = [0]*(len(features_parameter))
        for i in range(len(outputp)):
            for k in range(len(temp_parameter)):
                if(k == 0):
                    temp_parameter[k] += (self.hypothesis(features_parameter,
                                                          inputp, i)-outputp[i])
                else:
                    temp_parameter[k] += (self.hypothesis(features_parameter,
                                                          inputp, i)-outputp[i])*inputp[i, k-1]
        for i in range(len(features_parameter)):
            features_parameter[i] -= ((alpha*temp_parameter[i])/len(outputp))
        return features_parameter

    def train_SGD(self, outputp, features_parameter, features, alpha, epochs):          # train and loss SGD
        cost_history = []
        validation_loss = []
        test_loss = []
        for i in range(epochs):
            features_parameters = self.sgd(
                outputp, features_parameter, features, alpha)
            cost = self.cost_function(outputp, features_parameter, features)
            validation_loss.append(self.cost_function(
                self.val_y, features_parameter, self.val_x))
            test_loss.append(self.cost_function(
                self.test_y, features_parameter, self.test_x))
            cost_history.append(cost)
        print(i+1, features_parameter, cost)
        return features_parameters, cost_history, validation_loss, test_loss

    def train_BGD(self, outputp, features_parameter, features, alpha, epochs):          #train  and loss BGD
        cost_history = []
        validation_loss = []
        test_loss = []
        for i in range(epochs):
            features_parameters = self.bgd(
                outputp, features_parameter, features, alpha)
            cost = self.cost_function(outputp, features_parameter, features)
            validation_loss.append(self.cost_function(
                self.val_y, features_parameter, self.val_x))
            test_loss.append(self.cost_function(
                self.test_y, features_parameter, self.test_x))
            cost_history.append(cost)
        print(i+1, features_parameter, cost)
        return features_parameters, cost_history, validation_loss, test_loss

    def fit(self, x, y):                                                # fri
        training_loss_SGD = []
        training_loss_BGD = []
        validate_loss_SGD = []
        validate_loss_BGD = []
        test_loss_SGD = []
        test_loss_BGD = []
        self.features_parameter_SGD = [0]*(len(x[0])+1)
        self.features_parameter_BGD = [0]*(len(x[0])+1)
        self.features_parameter_SGD, training_loss_SGD, validate_loss_SGD, test_loss_SGD = self.train_SGD(
            y, self.features_parameter_SGD, x, self.alpha_SGD, self.ite_SGD)
        self.features_parameter_BGD, training_loss_BGD, validate_loss_BGD, test_loss_BGD = self.train_BGD(
            y, self.features_parameter_BGD, x, self.alpha_BGD, self.ite_BGD)
        return training_loss_SGD, training_loss_BGD, validate_loss_SGD, validate_loss_BGD, test_loss_SGD, test_loss_BGD
    def accuracy(self,x,y):                                                      # accuracy
        # print(x,y)
        sgd,bgd=self.predict(x)
        asgd=(np.count_nonzero(sgd == y)/len(sgd)*100)
        abgd=(np.count_nonzero(bgd==y)/len(bgd)*100)
        return asgd,abgd
        
    def predict(self, x):                                                   # predict
        sgd_predict=[]
        for i in range(len(x)):
            temp=self.hypothesis(self.features_parameter_SGD,x,i)
            if(temp<=0.5):
                sgd_predict.append(0)
            elif(temp>0.5):
                sgd_predict.append(1)
        bgd_predict=[]
        for i in range(len(x)):
            temp=self.hypothesis(self.features_parameter_BGD,x,i)
            if(temp<=0.5):
                bgd_predict.append(0)
            elif(temp>0.5):
                bgd_predict.append(1)
        return sgd_predict,bgd_predict