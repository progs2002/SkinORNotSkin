import numpy as np

def trainer_init(X,Y):
    print("initializing trainer")
    n = 4
    one = np.ones(245057,dtype=int)
    x_matrix = np.column_stack((one,X[:,0]))
    x_matrix = np.column_stack((x_matrix,X[:,1]))
    x_matrix = np.column_stack((x_matrix,X[:,2]))
    theta = np.zeros(3,detype=int)

    
