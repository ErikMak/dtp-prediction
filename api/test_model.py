from model import Model
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

@pytest.fixture
def mock_data_row():
    data = {'exp':[43],'peop_in_car':[1],'vehicle_age':[15],'dayofweek_sin':[0.974928],
            'dayofweek_cos':[-0.222521],'month_sin':[1.0],'month_cos':[6.123234e-17],'hour_sin':[-0.866025],
            'hour_cos':[0.5],'light':[0],'gender':[0],'rain':[1],'snow':[0],'cloudy':[1],'speed':[1],
            'low_distance':[0],'op_way':[0],'give_way':[0],'pedestrian_cross':[0],'houses':[0],'yard_exit':[1],
            'r_roadcross':[0],'nr_roadcross':[0],'school':[0],'bridge':[0],'industry':[0],'mall':[0],'bus_stop':[0],
            'roundabout':[0],'administration':[0],'rwd':[0],'awd':[0],'fwd':[1],'weight':[1500],'power':[150],
            'village':[0],'city':[0],'highway':[1],'cluster':[1],'medium':[1],'spacious':[0],
            'heavy': [0],'small': [0]}
    return pd.DataFrame(data)

@pytest.fixture
def mock_gray_image():
    try:
        M_image = np.load('./res/m_image.npy') 
        if(not np.allclose(M_image[0][10][8], np.array([134.02415579]), atol=1e-8)):
            raise ValueError
        return M_image
    except ValueError:
        print('Некорректное изображение M_image!')

def test_load_model_failure():
    with patch('tensorflow.keras.models.load_model', side_effect=ValueError):
        assert Model('tasp_cnn.h5', 'scaler.pkl', 'feature_weights.npy'), '[!] Модель нейронной сети не была загружена.' 

def test_load_scaler_failure():
    with patch('joblib.load',  side_effect=FileNotFoundError):
        assert Model('tasp_cnn.h5', 'scaler.pkl', 'feature_weights.npy'), '[!] Файл нормализатора не был загружен.' 

def test_load_weights_failure():
    with patch('numpy.load', side_effect=FileNotFoundError):
        assert Model('tasp_cnn.h5', 'scaler.pkl', 'feature_weights.npy'), '[!] Файл весов признаков не был загружен.'

def test_load_model_success():
    model = Model('tasp_cnn.h5', 'scaler.pkl', 'feature_weights.npy')
    assert model.model != None

def test_load_scaler_success():
    model = Model('tasp_cnn.h5', 'scaler.pkl', 'feature_weights.npy')
    assert model.scaler != None

def test_load_weights_success():
    model = Model('tasp_cnn.h5', 'scaler.pkl', 'feature_weights.npy')
    assert model.feature_weights.shape == (43,)

def test_correct_scaler(mock_data_row):
    model = Model('tasp_cnn.h5', 'scaler.pkl', 'feature_weights.npy')
    scaler = model.get_scaler()

    a = np.array([ 2.54236231, -0.48209734,  0.54290158,  1.34948922, -0.27257952,
        1.59596034,  0.01556593, -0.90430821,  1.27601914, -0.56593991,
       -0.47508625,  1.5067293 , -0.40784863,  1.14133258,  3.54195632,
       -0.44856356, -0.25881447, -0.6942425 , -0.67771234, -1.09264131,
        3.26025586, -0.52622698, -0.82011658, -0.42472674, -0.28720771,
       -0.30095913, -0.40808179, -0.69933492, -0.15697718, -0.53380175,
       -0.42998913, -0.38971311,  0.6419061 , -0.27793872,  0.05722103,
       -0.16340477, -1.5182196 ,  1.80335267, -0.70009979,  0.49969412,
       -0.35713047, -0.25260799, -0.16564597])
    
    b = scaler.transform(mock_data_row)

    assert np.allclose(a, b[0], atol=1e-8)

def test_correct_FV_matrix():
    model = Model('tasp_cnn.h5', 'scaler.pkl', 'feature_weights.npy')
    FV = model.get_FV()

    assert len(FV) == 6
    assert len(FV[3]) == 16
    assert FV[0][0] == 0.10702022406518259
    assert FV[1][10] == 0.0017532373053979339

def test_correct_weights():
    model = Model('tasp_cnn.h5', 'scaler.pkl', 'feature_weights.npy')
    feature_weights = model.get_weights()

    assert len(feature_weights) == 43
    assert feature_weights[0] == 0.10702022406518259
    assert feature_weights[42] == 0.0017532373053979339

def test_model_prediction(mock_gray_image):
    model = Model('tasp_cnn.h5', 'scaler.pkl', 'feature_weights.npy')

    tasp_cnn = model.get_model()
    result = tasp_cnn.predict(mock_gray_image)
    predicted_class = np.argmax(result, axis=1)

    assert int(predicted_class[0]) == 1