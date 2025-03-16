from objects import Objects
from unittest.mock import patch
from pyogrio.errors import DataSourceError
import pytest

@pytest.fixture
def mock_coords():
    return 58.546830, 31.298021

def test_load_model_success():
    ob = Objects()
    assert not ob._gdf.empty

def test_load_coordinates_failure():
    with patch('geopandas.read_file', side_effect=DataSourceError):
        assert Objects(), '[!] Карта геокодирования не была загружена.'

def test_find_objects_nearby(mock_coords):
    lat, long = mock_coords

    ob = Objects()
    gdf = ob._find_objects_nearby(lat, long)
    assert gdf['type'].count() == 6
    assert gdf[gdf['type'].eq("pedestrian_cross")]['type'].count() == 1
    assert gdf[gdf['type'].eq("mall")]['type'].count() == 1

def test_objects_nearby(mock_coords):
    lat, long = mock_coords

    ob = Objects()
    od = ob.get_objects_nearby(lat, long)
    od = od.model_dump()

    assert od["pedestrian_cross"] == 1
    assert od["mall"] == 1
    assert od["bridge"] == 0