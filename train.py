import numpy as np 
import pandas as pd 

data = pd.read_csv('/home/boostdvd/workspace/neural_network/data/mnist_train.csv')
data = np.array(data)
data = data.T
y_train = data[0] # wyniki dla zdjec
x_train = data[1:] # zdjecia
x_train = x_train / 255.0


def weights_and_bias():
    W1 = np.random.rand(128, 784) - 0.5
    b1 = np.random.rand(128, 1) - 0.5
    W2 = np.random.rand(10, 128) - 0.5
    b2 = np.random.rand(10, 1) - 0.5

    return W1, b1, W2, b2

def relu(Z):
    return np.maximum(0, Z)
    
def softmax(Z):
    return np.exp(Z) / np.sum(np.exp(Z), axis=0)
    
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = relu(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)

    return Z1, A1, Z2, A2

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    return one_hot_Y.T

def deriv_relu(Z):
    return Z > 0

def backward_prop(Z1, A1, A2, W2, X, Y):
    m = Y.size 
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * deriv_relu(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = weights_and_bias()
    
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, A2, W2, X, Y)

        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        
        if i % 10 == 0:
            print("Epoka: ", i)
            predictions = get_predictions(A2)
            print("Skuteczność (Accuracy):", round(get_accuracy(predictions, Y) * 100, 2), "%")
            print("-------------------------")
            
    return W1, b1, W2, b2

if __name__ == "__main__":
    W1, b1, W2, b2 = gradient_descent(x_train, y_train, 0.10, 500)
    np.savez('data/model_weights.npz', W1=W1, b1=b1, W2=W2, b2=b2)