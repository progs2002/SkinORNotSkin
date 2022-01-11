import numpy as np

def fixlabel(y):
    if y == 2: 
        return 0
    else:
        return 1

def extract():
    print("extracting dataset.......")
    dataset = np.loadtxt('Skin_NonSkin.txt',dtype=int)
    X, Y = dataset[:,0:3], dataset[:,3]
    for i in range(len(Y)):
        Y[i] = fixlabel(Y[i])
    print("dataset extracted")
    return X, Y

def distance(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)

r, g, b = map(int,input('Enter the R G B colour values seperated by a space\n').split())

query = np.array([b, g, r])

X, Y = extract()
print("prediction in progress")
entries=[]
for i in range(X.shape[0]):
    entries.append((i,distance(query,X[i,:]),X[i,:],Y[i]))
entries.sort(key=lambda y: y[1])

K = 30 
sum = 0
for i in range(0,K):
    sum += entries[i][3]
if sum >= K/2:
    print("(%d,%d,%d) is a valid skin colour"%(r,g,b))
else:
    print("(%d,%d,%d) is not a valid skin colour"%(r,g,b))