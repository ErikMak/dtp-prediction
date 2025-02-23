import numpy as np
from model import Model
from data_model import Data

class Predictor:
    def __init__(self):
        self.model = Model('model.pkl', 'features.pickle')

    def predict(self, data: Data):
        data_dict = data.dict()
        features = self.model.get_features()
        to_predict = [data_dict[feature] for feature in features]

        to_predict = np.array(to_predict)
        prediction = self.model.get_model().predict(to_predict.reshape(1, -1))

        return {"prediction": int(prediction[0])}