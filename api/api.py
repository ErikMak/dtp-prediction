from fastapi import FastAPI
from predictor import Predictor
from pydantic_data import data_model
import uvicorn

app = FastAPI(title="ML API система прогнозирования серьезности ДТП")
predictor = Predictor()

@app.post("/predict")
def predict(data: data_model.Data):
    return predictor.predict(data)

@app.get("/")
async def home():
    return {"message": "ML API система прогнозирования серьезности ДТП"}