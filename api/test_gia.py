from gia import GIA
import pytest
import numpy as np
import pandas as pd
from model import Model

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

@pytest.fixture
def mock_FM():
    FM = [[1, 2, 3, 4, 5],
      [6, 7, 8],
      [9, 10, 11, 12]]
    
    return FM

@pytest.fixture
def mock_FV():
    FV = [[0.03, 0.06, 0.01, 0.11, 0.15],
        [0.1, 0.04, 0.3],
        [0.02, 0.08, 0.8, 0.07]]
    
    return FV

def test_get_image(mock_FV, mock_FM):
    gia = GIA()

    image = gia.get_image(mock_FV, mock_FM)

    m = np.ndarray(shape=(5,5), buffer=np.array(
        [[  0.  ,   0.  ,   0.  ,   0.  ,   0.  ],
       [  0.  , 127.5 , 170.  , 148.75,   0.  ],
       [191.25, 212.5 , 233.75, 255.  ,   0.  ],
       [ 21.25,  85.  , 106.25,  42.5 ,  63.75],
       [  0.  ,   0.  ,   0.  ,   0.  ,   0.  ]]
    ))

    assert image.shape == (5,5)
    assert np.array_equal(image, m)

def test_convert(mock_data_row, mock_gray_image):
    model = Model('tasp_cnn.h5', 'scaler.pkl', 'feature_weights.npy')
    scaler = model.get_scaler()
    gia = GIA()

    b = scaler.transform(mock_data_row)
    M_image = gia.convert(b, model)

    assert np.allclose(M_image, mock_gray_image, atol=1e-8)