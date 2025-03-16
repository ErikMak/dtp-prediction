import geopandas as gpd
from shapely.geometry import Point
from colorama import Fore
from geopandas import GeoDataFrame
from pyogrio.errors import DataSourceError
from pydantic_data.objects_model import ObjectsModel


class Objects:
    _gdf = None
    _RADIUS = 100

    def __init__(self):
        try: 
            self._gdf = gpd.read_file("./res/map.geojson")
            self._gdf.sindex
            self._gdf = self._gdf.to_crs(epsg=32633)
        except DataSourceError:
            print(Fore.RED + '[!]' + Fore.RESET + ' Карта геокодирования не была загружена.')

    def _find_objects_nearby(self, lat: float, long: float) -> GeoDataFrame:
        """
        Ищет объекты поблизости в радиусе 100 метров
        
        :param lat: Широта - число с плавающей точкой
        :param long: Долгота - число с плавающей точкой
        :return: GeoDataFrame объектов поблизости
        """
        p = gpd.GeoDataFrame(
            geometry=[Point(long, lat)], crs="EPSG:4326"
        ).to_crs(self._gdf.crs).geometry[0]
        nearby_objects = self._gdf[self._gdf.distance(p) <= self._RADIUS]
        return nearby_objects
    
    def get_objects_nearby(self, lat: float, long: float) -> ObjectsModel:
        """
        Возвращает объекты поблизости в радиусе 100 метров
        
        :param lat: Широта - число с плавающей точкой
        :param long: Долгота - число с плавающей точкой
        :return: Модель Pydantic ObjectsModel
        """
        nearby_objects = self._find_objects_nearby(lat, long)
        has_pedestrian_cross = any(nearby_objects['type'] == 'pedestrian_cross')
        has_houses = any(nearby_objects['type'] == 'houses')
        has_yard_exit = any(nearby_objects['type'] == 'yard_exit')
        has_mall = any(nearby_objects['type'] == 'mall')
        has_r_roadcross = any(nearby_objects['type'] == 'r_roadcross')
        has_nr_roadcross = any(nearby_objects['type'] == 'nr_roadcross')
        has_school = any(nearby_objects['type'] == 'school')
        has_bridge = any(nearby_objects['type'] == 'bridge')
        has_administration = any(nearby_objects['type'] == 'administration')
        has_roundbaout = any(nearby_objects['type'] == 'roundbaout')
        has_industry = any(nearby_objects['type'] == 'industry')
        has_bus_stop = any(nearby_objects['type'] == 'bus_stop')
        has_highway = any(nearby_objects['type'] == 'highway')

        return ObjectsModel(
            pedestrian_cross=has_pedestrian_cross,
            houses=has_houses,
            yard_exit=has_yard_exit,
            r_roadcross=has_r_roadcross,
            nr_roadcross=has_nr_roadcross,
            school=has_school,
            bridge=has_bridge,
            industry=has_industry,
            roundabout=has_roundbaout,
            mall=has_mall,
            bus_stop=has_bus_stop,
            administration=has_administration,
            highway=has_highway,
        )
        