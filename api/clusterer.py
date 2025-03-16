import joblib
import pandas as pd
import numpy as np
from numpy import ndarray
from sklearn.metrics.pairwise import haversine_distances
from colorama import Fore

class Clusterer:
    df = None
    db = None

    def __init__(self):
        try:  
            self.db = joblib.load('./res/dbscan_model.pkl')
        except FileNotFoundError:
            print(Fore.RED + '[!]' + Fore.RESET + ' Файл кластаризации dbscan не был загружен.')
        try: 
            self.df = pd.read_csv('./res/coordinates.csv')
        except FileNotFoundError:
            print(Fore.RED + '[!]' + Fore.RESET + ' Файл координат для кластеризации не был загружен.')

    def _haversine(self, coords: ndarray) -> ndarray:
        """
        Вычисляет матрицу расстояний между всеми парами точек по формуле гаверсинуса.
        
        :param coords: Массив координат в формате [[lat1, lon1], [lat2, lon2], ...]
        :return: Матрица расстояний в радианах
        """
        # Преобразуем координаты в радианы
        coords_rad = np.radians(coords)
        
        # Вычисляем матрицу расстояний
        dist_matrix = haversine_distances(coords_rad)
        
        return dist_matrix

    def predict(self, lat: float, long: float) -> int:
        """
        Назначает координате номер кластера
        
        :param lat: Широта - число с плавающей точкой
        :param long: Долгота - число с плавающей точкой
        :return: Целочисленный номер кластера
        """
        new_point_df = pd.DataFrame({'lat': [lat], 'long': [long]})
    
        M = pd.concat([self.df, new_point_df], ignore_index=True)
        coords = M[['lat', 'long']].values

        dist_matrix = self._haversine(coords)
        clusters = self.db.fit_predict(dist_matrix)
        M['cluster'] = clusters

        return int(M.iloc[-1]['cluster'])