from fastapi import FastAPI
from joblib import load
from sklearn.svm import SVC

model = load('svc_model.joblib')

app = FastAPI()

@app.get("/")
async def root():
    return {'message':'Welcome to the Skin or NotSkin'}

@app.get("/predict")
def predict(R:int,G:int,B:int):
    x = [[R,G,B]]
    out = model.predict(x)
    if out == 0:
        pred = 'not skin'
    else:
        pred = 'skin'
    return {'prediction':pred, 'class':pred}
