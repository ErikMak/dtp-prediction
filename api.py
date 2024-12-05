import pickle
import numpy as np
from pydantic import BaseModel

from fastapi import FastAPI

app = FastAPI()


clf = pickle.load(open('model.pkl', 'rb'))
features = pickle.load(open('features.pickle', 'rb'))

class Data(BaseModel):
    peop_in_car: int
    exp: int
    city: int
    highway: int
    gender: int
    speed: int
    op_way: int
    give_way: int
    red_light: int
    bad_grip: int
    bad_visibility: int
    houses: int
    r_roadcross: int
    nr_roadcross: int
    industry: int
    mall: int
    rwd: int
    awd: int
    fwd: int
    weight: int
    power: int
    vehicle_age: int
    month: int
    d_week: int
    time_of_day: int
    distance: float
    medium: int
    spacious: int
    heavy: int
    small: int
        
@app.post("/predict")
def predict(data: Data):
    data_dict = data.dict()
    to_predict = [data_dict[feature] for feature in features]

    to_predict = np.array(to_predict)
    
    prediction = clf.predict(to_predict.reshape(1, -1))
    
    return {"prediction": int(prediction[0])}