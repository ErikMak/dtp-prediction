from fastapi import FastAPI
from predictor import Predictor
from data_model import Data

app = FastAPI()
predictor = Predictor()

@app.post("/predict")
def predict(data: Data):
    return predictor.predict(data)