import numpy as np
import plotter
#import trainer
from scipy.special import expit
import sys
import os

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

def h2(x, weights):
    return expit(np.dot(x,weights))

def h(x, weights):
    matrix = (x @ weights)
    for i in range(matrix.shape[0]):
        matrix[i] = expit(matrix[i])
    return matrix

def cost_function(x, y, weights):
    m = y.shape[0]
    cost1=0
    cost2=0
    for i in range(m):
        if y[i] == 1:
            cost1 += y[i] * np.log(h2(x[i,:], weights))
        else:
            cost2 += (1 - y[i]) * np.log(1 - h2(x[i,:], weights))
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
        #cost_history.append(cost_function(x, y, weights))
    return weights

def classifier(features, weight):
    prediction = expit(np.dot(features, weight))
    if prediction >= 0.5:
        return True
    else:
        return False

def train_init():
    X, Y = prepare_trainer()

    w_temp = np.array([1,1,1,1])

    alpha = 0.3
    iterations = 5000

    weights = train(X, Y, w_temp, alpha, iterations)
    print("training done")
    print("weights are ")
    print(weights)
    np.savetxt('weights.txt',weights)

def predict():
    if os.path.exists('weights.txt') == False:
        print("model has not been trained yet")
        sys.exit(0)
    w = np.loadtxt('weights.txt')
    x = map(float,input().split())
    x = list(x)
    x.insert(0,1)
    x = np.array(x)
    if classifier(x,w):
        print(x[1:4,].__str__(), 'is a valid skin color')
    else:
        print(x[1:4,].__str__(), 'is not a valid skin color')

if __name__ == "__main__":
    if len(sys.argv) == 2:
        if sys.argv[1] == 'train':
            train_init()
            print("model has been trained, the weights have been saved as weights.txt")
        elif sys.argv[1] == 'predict':
            predict()
        else:
            print("invlaid argument")

