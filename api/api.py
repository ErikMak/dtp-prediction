from fastapi import FastAPI
from predictor import Predictor
from pydantic_data import data_model
import uvicorn

app = FastAPI()
predictor = Predictor()

@app.post("/predict")
def predict(data: data_model.Data):
    return predictor.predict(data)


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)