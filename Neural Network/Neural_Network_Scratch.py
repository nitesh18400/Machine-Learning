# Neural Netwoek Implementation with all types of activation and function with batch Size .
import numpy as np
import pickle

class layer():
    def __init__(self, layer_left, layer_right, layer_activation, learning_rate, weight_init_method):
        self.left = layer_left
        self.right = layer_right
        self.activation = layer_activation
        self.learning_rate = learning_rate
        self.weight_init_method = weight_init_method
        tup = (self.right, self.left)
        if(weight_init_method == "zero"):
            self.weight = self.zero_init(tup)
        if(weight_init_method == "random"):
            self.weight = self.random_init(tup)
        if(weight_init_method == "normal"):
            self.weight = self.normal_init(tup)
        bias_tup = (self.right, 1)
        self.bias = self.bias_init(bias_tup)

    def relu(self, X):
        """
        Calculating the ReLU activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        Z=np.copy(X)
        Z[Z<0] = Z[Z<0]*0.01
        return Z

    def relu_grad(self, X):
        """
        Calculating the gradient of ReLU activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        dx = np.ones_like(X)
        dx[X < 0] = 0.01
        return dx

    def sigmoid(self, X):
        """
        Calculating the Sigmoid activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        return 1/(1+np.exp(-X))

    def sigmoid_grad(self, X):
        """
        Calculating the gradient of Sigmoid activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        return self.sigmoid(X)*(1-self.sigmoid(X))

    def linear(self, X):
        """
        Calculating the Linear activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        return X

    def linear_grad(self, X):
        """
        Calculating the gradient of Linear activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        ans=np.zeros(shape=X.shape)
        ans=ans+1
        return ans

    def tanh(self, X):
        """
        Calculating the Tanh activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        return np.tanh(X)

    def tanh_grad(self, X):
        """
        Calculating the gradient of Tanh activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        return 1-((self.tanh(X))**2)

    def softmax(self, X):
        """
        Calculating the ReLU activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        return np.exp(X-X.max())

    def softmax_grad(self, X):
        """
        Calculating the gradient of Softmax activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        return self.softmax(X)/np.sum(self.softmax(X), axis=0) * (1-self.softmax(X)/np.sum(self.softmax(X), axis=0))

    def zero_init(self, shape):
        """
        Calculating the initial weights after Zero Activation for a particular layer

        Parameters
        ----------
        shape : tuple specifying the shape of the layer for which weights have to be generated 

        Returns
        -------
        weight : 2-dimensional numpy array which contains the initial weights for the requested layer
        """
        weight = np.zeros(shape=(shape[0], shape[1]))
        return weight

    def random_init(self, shape):
        """
        Calculating the initial weights after Random Activation for a particular layer

        Parameters
        ----------
        shape : tuple specifying the shape of the layer for which weights have to be generated 

        Returns
        -------
        weight : 2-dimensional numpy array which contains the initial weights for the requested layer
        """
        weight = np.random.randn(shape[0], shape[1]) * 0.01
        return weight

    def normal_init(self, shape):
        """
        Calculating the initial weights after Normal(0,1) Activation for a particular layer

        Parameters
        ----------
        shape : tuple specifying the shape of the layer for which weights have to be generated 

        Returns
        -------
        weight : 2-dimensional numpy array which contains the initial weights for the requested layer
        """
        weight = np.random.normal(0, 1, shape)*0.01
        return weight

    def bias_init(self, shape):
        return np.zeros(shape=(shape[0], shape[1]))

    def update_weight_bias(self, dW, dB):
        self.weight = self.weight-self.learning_rate*dW
        self.bias = self.bias-self.learning_rate*dB
        return None
    def feedforward_for_validation(self, input):
        A_prev = input
        Z = np.dot(self.weight,A_prev)+self.bias
        if(self.activation == "relu"):
            A = self.relu(Z)
        if(self.activation == "sigmoid"):
            A = self.sigmoid(Z)
        if(self.activation == "linear"):
            A = self.linear(Z)
        if(self.activation == "tanh"):
            A = self.tanh(Z)
        if(self.activation == "softmax"):
            A = self.softmax(Z)
        return A
    

    def feedforward(self, input):
        self.A_prev = input
        self.Z = np.dot(self.weight, self.A_prev)+self.bias
        if(self.activation == "relu"):
            self.A = self.relu(self.Z)
        if(self.activation == "sigmoid"):
            self.A = self.sigmoid(self.Z)
        if(self.activation == "linear"):
            self.A = self.linear(self.Z)
        if(self.activation == "tanh"):
            self.A = self.tanh(self.Z)
        if(self.activation == "softmax"):
            self.A = self.softmax(self.Z)
        return self.A

    def backward(self, dA):
        dZ = []
        if(self.activation == "relu"):
            dZ = np.multiply(self.relu_grad(self.Z), dA)
        if(self.activation == "sigmoid"):
            dZ = np.multiply(self.sigmoid_grad(self.Z), dA)
        if(self.activation == "linear"):
            dZ = np.multiply(self.linear_grad(self.Z), dA)
        if(self.activation == "tanh"):
            dZ = np.multiply(self.tanh_grad(self.Z), dA)
        if(self.activation == "softmax"):
            dZ = np.multiply(self.softmax_grad(self.Z), dA)

        dW = (np.dot(dZ, self.A_prev.T))/dZ.shape[1]
        db = (np.sum(dZ, axis=1, keepdims=True))/dZ.shape[1]
        dA_prev = np.dot(self.weight.T, dZ)

        self.update_weight_bias(dW=dW, dB=db)
        return dA_prev
class MyNeuralNetwork():
    """
    My implementation of a Neural Network Classifier.
    """
    layer = []
    cost = []
    val_cost=[]
    vis_A=[]
    acti_fns = ['relu', 'sigmoid', 'linear', 'tanh', 'softmax']
    weight_inits = ['zero', 'random', 'normal']

    def __init__(self, n_layers=5, layer_sizes=[784, 256, 128, 64, 10], activation="sigmoid", learning_rate=0.1, weight_init_method="normal", batch_size=100, num_epochs=100):
        """
        Initializing a new MyNeuralNetwork object

        Parameters
        ----------
        n_layers : int value specifying the number of layers

        layer_sizes : integer array of size n_layers specifying the number of nodes in each layer

        activation : string specifying the activation function to be used
                     possible inputs: relu, sigmoid, linear, tanh

        learning_rate : float value specifying the learning rate to be used

        weight_init : string specifying the weight initialization function to be used
                      possible inputs: zero, random, normal

        batch_size : int value specifying the batch size to be used

        num_epochs : int value specifying the number of epochs to be used
        """
        self.n_layers = n_layers
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.learning_rate = learning_rate
        self.weight_init_method = weight_init_method
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        if activation not in self.acti_fns:
            raise Exception('Incorrect Activation Function')

        if weight_init_method not in self.weight_inits:
            raise Exception('Incorrect Weight Initialization Function')
        pass
    def relu(self, X):
        """
        Calculating the ReLU activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        Z=np.copy(X)
        Z[Z<0] = Z[Z<0]*0.01
        return Z

    def relu_grad(self, X):
        """
        Calculating the gradient of ReLU activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        dx = np.ones_like(X)
        dx[X < 0] = 0.01
        return dx

    def sigmoid(self, X):
        """
        Calculating the Sigmoid activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        return 1/(1+np.exp(-X))

    def sigmoid_grad(self, X):
        """
        Calculating the gradient of Sigmoid activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        return self.sigmoid(X)*(1-self.sigmoid(X))

    def linear(self, X):
        """
        Calculating the Linear activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        return X

    def linear_grad(self, X):
        """
        Calculating the gradient of Linear activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        ans=np.zeros(shape=X.shape)
        ans=ans+1
        return ans

    def tanh(self, X):
        """
        Calculating the Tanh activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        return np.tanh(X)

    def tanh_grad(self, X):
        """
        Calculating the gradient of Tanh activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        return 1-((self.tanh(X))**2)

    def softmax(self, X):
        """
        Calculating the ReLU activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        return np.exp(X-X.max())

    def softmax_grad(self, X):
        """
        Calculating the gradient of Softmax activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        return self.softmax(X)/np.sum(self.softmax(X), axis=0) * (1-self.softmax(X)/np.sum(self.softmax(X), axis=0))

    def zero_init(self, shape):
        """
        Calculating the initial weights after Zero Activation for a particular layer

        Parameters
        ----------
        shape : tuple specifying the shape of the layer for which weights have to be generated 

        Returns
        -------
        weight : 2-dimensional numpy array which contains the initial weights for the requested layer
        """
        weight = np.zeros(shape=(shape[0], shape[1]))
        return weight

    def random_init(self, shape):
        """
        Calculating the initial weights after Random Activation for a particular layer

        Parameters
        ----------
        shape : tuple specifying the shape of the layer for which weights have to be generated 

        Returns
        -------
        weight : 2-dimensional numpy array which contains the initial weights for the requested layer
        """
        weight = np.random.randn(shape[0], shape[1]) * 0.01
        return weight

    def normal_init(self, shape):
        """
        Calculating the initial weights after Normal(0,1) Activation for a particular layer

        Parameters
        ----------
        shape : tuple specifying the shape of the layer for which weights have to be generated 

        Returns
        -------
        weight : 2-dimensional numpy array which contains the initial weights for the requested layer
        """
        weight = np.random.normal(0, 1, shape)*0.01
        return weight
    def cross_entropy_loss(self, y, a):
        N = a.shape[0]
        ce = (-np.sum(y * np.log(a))) / N
        return ce

    def cross_entropy_loss_grad(self, y, a):
        return a-y

    def weight_bias_intial(self):
        return None

    def feedforward(self, input_layer):
        return None

    def layer_formation(self):
        for i in range(len(self.layer_sizes)-1):
            if(len(self.layer_sizes)-1==i+1):
              self.layer.append(layer(layer_left=self.layer_sizes[i], layer_right=self.layer_sizes[i+1],layer_activation="sigmoid", learning_rate=self.learning_rate, weight_init_method=self.weight_init_method))
            else:
              self.layer.append(layer(layer_left=self.layer_sizes[i], layer_right=self.layer_sizes[i+1],layer_activation=self.activation, learning_rate=self.learning_rate, weight_init_method=self.weight_init_method))
        return self.layer

    def fit(self, X, y,X_val,y_val):
        """
        Fitting (training) the linear model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as training data.

        y : 1-dimensional numpy array of shape (n_samples,) which acts as training labels.

        Returns
        -------
        self : an instance of self
        """
        self.layer_formation()
        for iter in range(self.num_epochs):
            for batch in range(X.shape[0]//self.batch_size):
              X_curr=X[batch*self.batch_size:(batch+1)*self.batch_size,:]
              y_curr=y[batch*self.batch_size:(batch+1)*self.batch_size,:]
              A = X_curr.T
              for l in self.layer:
                  A = l.feedforward(A)
              curr_cost = np.sum(self.cross_entropy_loss(y_curr.T, A))/X_curr.T.shape[1]
              dA = self.cross_entropy_loss_grad(y_curr.T, A)
              for l in reversed(self.layer):
                  dA = l.backward(dA)
            A_val=X_val.T
            for l in self.layer:
                  A_val=l.feedforward_for_validation(A_val)
            self.val_cost.append(np.sum(self.cross_entropy_loss(y_val.T, A_val))/X_val.T.shape[1])
            self.cost.append(curr_cost)
            print(iter)
        self.vis_A=X.T
        for l in range(len(self.layer)-1):
                  self.vis_A = self.layer[l].feedforward(self.vis_A)
        # fit function has to return an instance of itself or else it won't work with test.py
        return self

    def predict_proba(self, X):
        """
        Predicting probabilities using the trained linear model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as testing data.

        Returns
        -------
        y : 2-dimensional numpy array of shape (n_samples, n_classes) which contains the 
            class wise prediction probabilities.
        """

        # return the numpy array y which contains the predicted values
        A = X.T
        for l in self.layer:
            A = l.feedforward(A)
        return A

    def predict(self, X):
        """
        Predicting values using the trained linear model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as testing data.

        Returns
        -------
        y : 1-dimensional numpy array of shape (n_samples,) which contains the predicted values.
        """

        # return the numpy array y which contains the predicted values
        A = X.T
        for l in self.layer:
            A = l.feedforward(A)
        # print(A)
        A = A.T
        pred = []
        for i in range(len(A)):
            pred.append(np.argmax(A[i]))
        return pred

    def score(self, X, y):
        """
        Predicting values using the trained linear model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as testing data.

        y : 1-dimensional numpy array of shape (n_samples,) which acts as testing labels.

        Returns
        -------
        acc : float value specifying the accuracy of the model on the provided testing set
        """

        # return the numpy array y which contains the predicted values
        acc = 0
        pred = self.predict(X)
        for i in range(len(y)):
            if(pred[i] == np.argmax(y[i])):
                acc += 1
        return (acc/len(y))
