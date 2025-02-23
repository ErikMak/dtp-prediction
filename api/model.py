import pickle

class Model:
    def __init__(self, model_path: str, features_path: str):
        self.clf = pickle.load(open(model_path, 'rb'))
        self.features = pickle.load(open(features_path, 'rb'))

    def get_model(self):
        return self.clf

    def get_features(self):
        return self.features