import matplotlib.pyplot as plt

def showplot(X, Y):
    ax = plt.axes(projection="3d")
    skin = X[0:50859,:]
    nonskin = X[50859:,:]
    ax.scatter(skin[:,0],skin[:,1],skin[:,2],marker="o",label="skin")
    ax.scatter(nonskin[:,0],nonskin[:,1],nonskin[:,2],marker="^",label="nonskin")
    ax.legend()

    plt.show()