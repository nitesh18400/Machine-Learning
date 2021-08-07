import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
class MyLinearRegression():
	features_parameter_MAE = []                 #theta list      
	features_parameter_RMSE = []
	alpha_mae = 0                               #variou parameter for prediction                                        
	ite_mae = 0
	alpha_rmse = 0
	ite_rmse = 0
	test_x = pd.DataFrame()                       #required for loss function
	test_y = pd.DataFrame()

	def __init__(self):
		pass

	def learning_rate_iter(self, alpha_mae, ite_mae, alpha_rmse, ite_rmse):    #construcutoe
		self.alpha_mae = alpha_mae
		self.ite_mae = ite_mae
		self.alpha_rmse = alpha_rmse
		self.ite_rmse = ite_rmse

	def hypothesis(self, features_parameter, features, row_number):            #hypothesis
		h = 0
		for i in range(len(features_parameter)):
			if(i == 0):
				h += features_parameter[i]
			else:
				h += features_parameter[i]*features[row_number, i-1]
		return h

	def cost_function_MAE(self, outputp, features_parameter, features):          #mae cost function
		sumission = 0
		for i in range(len(outputp)):
			sumission = sumission + \
				abs(self.hypothesis(features_parameter,
									features, i)-outputp[i, 0])
		dividedby = len(outputp)
		return sumission/dividedby

	def cost_function_RMSE(self, outputp, features_parameter, features):        #rmse cost function
		sumission = 0
		for i in range(len(outputp)):
			sumission = sumission + \
				((self.hypothesis(features_parameter,
								  features, i)-outputp[i, 0])**2)
		dividedby = len(outputp)
		x = (sumission/dividedby)
		return math.sqrt(x)

	def finding_theta_MAE(self, outputp, features_parameter, features, alpha):         #update theta of mae
		temp_parameter = [0]*len(features_parameter)
		c = 0
		for i in range(len(outputp)):
			for k in range(len(features_parameter)):
				if(k == 0):
					if(self.hypothesis(features_parameter, features, i) > outputp[i, 0]):
						temp_parameter[k] += 1
					elif(self.hypothesis(features_parameter, features, i) < outputp[i, 0]):
						temp_parameter[k] -= 1
				else:
					if(self.hypothesis(features_parameter, features, i) > outputp[i, 0]):
						temp_parameter[k] += (1*features[i, k-1])
					elif(self.hypothesis(features_parameter, features, i) < outputp[i, 0]):
						temp_parameter[k] -= (1*features[i, k-1])
		for i in range(len(features_parameter)):
			features_parameter[i] -= ((alpha*temp_parameter[i])/(len(outputp)))
		return features_parameter

	def finding_theta_RMSE(self, outputp, features_parameter, features, alpha):                   #update theta of rmse
		temp_parameter = []
		for i in range(len(features_parameter)):
			temp_parameter.append([0, 0])
		for i in range(len(outputp)):
			for k in range(len(features_parameter)):
				if(k == 0):
					num = (self.hypothesis(
						features_parameter, features, i)-outputp[i, 0])
					den = (self.hypothesis(features_parameter,
										   features, i)-outputp[i, 0])**2
					temp_parameter[k][0] += num
					temp_parameter[k][1] += den
				else:
					num = (self.hypothesis(features_parameter,
										   features, i)-outputp[i, 0])*features[i, k-1]
					den = (self.hypothesis(features_parameter,
										   features, i)-outputp[i, 0])**2
					temp_parameter[k][0] += num
					temp_parameter[k][1] += den
		for i in range(len(features_parameter)):
			features_parameter[i] -= (((alpha*temp_parameter[i][0]*(
				math.sqrt(len(outputp))))/(math.sqrt(temp_parameter[i][1])))/len(outputp))
		return features_parameter

	def train_MAE(self, outputp, features_parameter, features, alpha, epochs):                #update theta of train_MAE
		cost_history = []
		validate_cost = []
		for i in range(epochs):
			features_parameters = self.finding_theta_MAE(
				outputp, features_parameter, features, alpha)
			cost = self.cost_function_MAE(
				outputp, features_parameter, features)
			cost_history.append(cost)
			v_cost = self.cost_function_MAE(
				self.test_y, features_parameter, self.test_x)
			validate_cost.append(v_cost)
		print(i+1, cost,v_cost)
		return features_parameters, cost_history, validate_cost

	def train_RMSE(self, outputp, features_parameter, features, alpha, epochs):               #train fnuction
		cost_history = []
		validate_cost = []
		for i in range(epochs):
			features_parameters = self.finding_theta_RMSE(
				outputp, features_parameter, features, alpha)
			cost = self.cost_function_RMSE(
				outputp, features_parameter, features)
			v_cost = self.cost_function_RMSE(
				self.test_y, features_parameter, self.test_x)
			cost_history.append(cost)
			validate_cost.append(v_cost)
		print(i+1, cost,v_cost)
		return features_parameters, cost_history, validate_cost

	def fit(self, x, y):                                      #fit function
		training_loss_MAE = []
		training_loss_RMSE = []
		validate_loss_MAE = []
		validate_loss_RMSE = []
		self.features_parameter_MAE = [0]*(len(x[0])+1)
		self.features_parameter_RMSE = [0]*(len(x[0])+1)
		self.features_parameter_MAE, training_loss_MAE, validate_loss_MAE = self.train_MAE(
			y, self.features_parameter_MAE, x, self.alpha_mae, self.ite_mae)
		self.features_parameter_RMSE, training_loss_RMSE, validate_loss_RMSE = self.train_RMSE(
			y, self.features_parameter_RMSE, x, self.alpha_rmse, self.ite_rmse)
		return training_loss_MAE, training_loss_RMSE, validate_loss_MAE, validate_loss_RMSE

	def predict(self, x):                                             #predict
		predicted_output_MAE = []
		predicted_output_RMSE = []
		for i in range(len(x)):
			h = 0
			for j in range(len(self.features_parameter_MAE)):
				if(j == 0):
					h += self.features_parameter_MAE[j]
				else:
					h += (self.features_parameter_MAE[j]*x[i, j-1])
			predicted_output_MAE.append(h)
		for i in range(len(x)):
			h = 0
			for j in range(len(self.features_parameter_RMSE)):
				if(j == 0):
					h += self.features_parameter_RMSE[j]
				else:
					h += (self.features_parameter_RMSE[j]*x[i, j-1])
			predicted_output_RMSE.append(h)
		return predicted_output_MAE, predicted_output_RMSE

	def normal_equation(self, x, y,tx,ty):                                     #normal equation form
		
		oo = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
		too= np.linalg.inv(tx.T.dot(tx)).dot(tx.T).dot(ty)
		cost_mae_train=self.cost_function_MAE(y, oo, x)
		cost_mae_test=self.cost_function_MAE(ty, too, tx)
		print("Cost MAE Train",cost_mae_train)
		print("Cost MAE Test", cost_mae_test)
		return oo