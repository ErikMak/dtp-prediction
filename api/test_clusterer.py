from clusterer import Clusterer
import pytest
import pandas as pd
from unittest.mock import patch

@pytest.fixture
def mock_coordinates_df():
    data = {'lat': [10.0, 20.0], 'long': [30.0, 40.0]}
    return pd.DataFrame(data)

def test_load_dbscan_model_success():
    clusterer = Clusterer()
    assert clusterer.db != None

def test_load_coordinates_success():
    clusterer = Clusterer()
    assert not clusterer.df.empty

def test_load_coordinates_failure():
    with patch('joblib.load', side_effect=FileNotFoundError):
        assert Clusterer(), '[!] Файл координат для кластеризации не был загружен.'

def test_load_dbscan_model_failure():
    with patch('pandas.read_csv', side_effect=FileNotFoundError):
        assert Clusterer(), '[!] Файл кластаризации dbscan не был загружен.'

def test_haversine(mock_coordinates_df):
    clusterer = Clusterer()
    coords = mock_coordinates_df[['lat', 'long']].values
    dist_matrix = clusterer._haversine(coords)
    
    assert dist_matrix.shape == (2, 2)

def test_predict():
    clusterer = Clusterer()
    
    assert clusterer.predict(58.206988, 30.723095) == 2