import numpy as np

def extract(filename):
    print("loading dataset")
    dataset = np.loadtxt(filename,dtype=int)
    return dataset[:,0:3], dataset[:,3]
