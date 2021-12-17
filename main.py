import numpy as np
import plotter
import trainer
import sys

def extract(filename):
    print("loading dataset")
    dataset = np.loadtxt(filename,dtype=int)
    return dataset[:,0:3], dataset[:,3]

def fixlabel(y):
    if y == 2: 
        return 0
    else:
        return 1

def prepare_trainer():
    X, Y = extract('Skin_NonSkin.txt')
    print("dataset loaded with %d examples" % X.shape[0])

    #prepare X
    one = np.ones(245057,dtype=int)
    x_matrix = np.column_stack((one,X[:,0]))
    x_matrix = np.column_stack((x_matrix,X[:,1]))
    x_matrix = np.column_stack((x_matrix,X[:,2]))
    X = x_matrix

    #prepare Y
    for i in range(len(Y)):
        Y[i] = fixlabel(Y[i])

    print("X")
    print(X)
    print("Y")
    print(Y)
    return X, Y

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def h(x, weights):
    matrix = (x @ weights)
    for i in range(matrix.shape[0]):
        matrix[i] = sigmoid(matrix[i])
    return matrix

def cost_function(x, y, weights):
    m = y.shape[0]
    cost1 = y.T @ np.log(h(x, weights))
    cost2 = (np.ones(m) - y.T) @ np.log(np.ones(m) - h(x, weights))
    return (-1/m * (cost1 + cost2))

def update_weights(x, y, weights, learning_rate):
    m = y.shape[0]
    weights = weights - ((learning_rate/m) * (x.T @ (h(x,weights) - y)))
    return weights

def train(x, y, weights, learning_rate, iter):
    cost_history = []
    for i in range(iter):
        print(i)
        weights = update_weights(x, y, weights, learning_rate)
        cost_history.append(cost_function(x, y, weights))
    return weights, cost_history

def classifier(features, weight):
    prediction = sigmoid(np.dot(features, weight))
    if prediction >= 0.5:
        return True
    else:
        return False

if __name__ == "__main__":
    X, Y = prepare_trainer()

    w_temp = np.array([0,0,0,0])

    alpha = 0.3

    weights, history = train(X, Y, w_temp, alpha, 20)
    print("training done")
    print("weights are ")
    print(weights)
    print("cost history")
    print(history)

    mat = h(X, w_temp)
