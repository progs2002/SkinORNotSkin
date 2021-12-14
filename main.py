import numpy as np
import extractor
import plotter
import trainer

def train():
    X, Y = extractor.extract('Skin_NonSkin.txt')
    print("dataset loaded with %d examples" % X.shape[0])
    print("generating 3d plot for given data")
    plotter.showplot(X,Y)
    trainer.trainer_init(X,Y)

def predict(h):
    pass