import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
data =  pd.read_csv('train.csv')
data = np.array(data)
m ,n = data.shape
print(m)
print(n)
np.random.shuffle(data)
data_cv = data[0:1000].T
Y_cv = data_cv[0]
X_cv = data_cv[1:n]
X_cv = X_cv/255.
data_train = data[1001:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train/255.
hidden_layer_size = 25
num_labels = 10 
input_layer_size = 784
_,m_train = X_train.shape
"""def initializeRandWeight(L_in,L_out):
    epsilon_init = 0.12
    W = np.random.rand(L_out, 1 + L_in)*2*epsilon_init - epsilon_init
    return W
def nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,lambda1):
    Y = np.zeros(num_labels,m)
    for i in range(m)
        Y(y(i),i) = 1  
    X = np.concatenate((np.ones(m,1),X),axis=1)
    a2 = sigmoid(X @ Theta1.T)
    a2 = np.concatenate((ones(size(a2,1),1), a2), axis=1)
    a3 = sigmoid(a2 @ Theta2.T)
    J = (-1/m)*((np.sum(Y.T*log(a3)))+np.sum((1-Y).T*log(1-a3))) #FIGURE OUT SYNTAX
    Re = np.sum(Theta1(:, 2:end)**2, 1)+ np.sum(Theta2(:, 2:end)**2, 1) #FIGURE OUT SYNTAX
    J = J + lambda*Re/(2*m); #FIGURE OUT SYNTAX
    for i = 1:m
        a1 = X(i,:);
        z2 = a1*Theta1';
        a2 = [1 sigmoid(z2)];
        z3 = a2 @ Theta2.T
        a3 = sigmoid(z3)
        del_3 = a3 - Y(:,i).T
        z2 = [1 z2]
        del_2 = (del_3*Theta2).*sigmoidGradient(z2);
        del_2 = del_2(2:end);
        Theta1_grad = Theta1_grad + del_2'*a1;
        Theta2_grad = Theta2_grad + del_3'*a2;
    Theta1_grad(:,1) =  Theta1_grad(:,1)./m;
    Theta1_grad(:,2:end) = Theta1_grad(:,2:end)./m +lambda*Theta1(:,2:end)/m;
    Theta2_grad(:,2:end) = Theta2_grad(:,2:end)./m + lambda*Theta2(:,2:end)/m;
    Theta2_grad(:,1) = Theta2_grad(:,1)./m;
    grad = [Theta1_grad(:) ; Theta2_grad(:)];
"""
#Above was attempts to convert octave code(Which I had already done) to python code    
def initParameters():
    W1 = np.random.rand(hidden_layer_size, input_layer_size) - 0.5
    b1 = np.random.rand(hidden_layer_size, 1) - 0.5
    W2 = np.random.rand(num_labels,hidden_layer_size) - 0.5
    b2 = np.random.rand(num_labels, 1) - 0.5
    return W1, b1, W2, b2
def sigmoid(x):
    z = np.exp(-x)
    sig = 1 / (1 + z)
    return sig
def sigmoidDerivative(x):
    outcome = sigmoid(x)
    return outcome*(1-outcome)
def forwardProp(W1,b1,W2,b2,X):
    Z1 = np.dot(W1,X) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(W2,A1) + b2
    A2 = sigmoid(Z2)
    return Z1, A1, Z2, A2
def oneHot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y
def backwardProp(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = oneHot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * sigmoidDerivative(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2
def updateParams(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2
def getPredictions(A2):
    return np.argmax(A2, 0) 
def getAccuracy(predictions, Y):
    print(predictions)
    print(Y)
    return np.sum(predictions == Y) / Y.size
def gradientDescent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = initParameters()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forwardProp(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backwardProp(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = updateParams(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = getPredictions(A2)
            print(getAccuracy(predictions, Y))
    return W1, b1, W2, b2
def makePredictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forwardProp(W1, b1, W2, b2, X)
    predictions = getPredictions(A2)
    return predictions
def testPrediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = makePredictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()
W1, b1, W2, b2 = gradientDescent(X_train, Y_train, 0.10, 1000)
testPrediction(0, W1, b1, W2, b2)
testPrediction(1, W1, b1, W2, b2)
testPrediction(2, W1, b1, W2, b2)
testPrediction(3, W1, b1, W2, b2)
dev_predictions = makePredictions(X_cv, W1, b1, W2, b2)
getAccuracy(dev_predictions, Y_cv)
