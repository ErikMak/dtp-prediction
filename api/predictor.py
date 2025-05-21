import numpy as np
from model import Model
from clusterer import Clusterer
from pydantic_data import data_model
from request.weather import Weather
from gia import GIA
from fastapi import HTTPException
from colorama import Fore
from objects import Objects
from pydantic_data import other_model
import pandas as pd
import concurrent.futures

class Predictor:
    model = None
    clusterer = None
    gia = None
    weather = None

    def __init__(self):
        # Модель нейронной сети
        self.model = Model('tasp_cnn.h5', 'scaler.pkl', 'feature_weights.npy')
        # Модуль кластеризации
        self.clusterer = Clusterer()
        # Модуль преобразования признаков в серые изображения 
        self.gia = GIA()
        # Модуль сбора погодных данных
        self.weather = Weather()
        # Модуль сбора геометок
        self.ob = Objects()

    def predict(self, data: data_model.Data):
        try:
            print(Fore.BLUE + '[!]' + Fore.RESET + f' lat: {data.lat};long: {data.long}')

            ### ПУЛ ПОТОКОВ
            with concurrent.futures.ThreadPoolExecutor() as executor:
                ### СБОР СТОРОННИМИ МОДУЛЯМИ НЕОБХОДИМЫХ ДАННЫХ
                # Сбор погодных данных и временной метки
                weather_future = executor.submit(self.weather.get_current_weather, data.lat, data.long)
                # Сбор данных о объектах поблизости
                objects_future = executor.submit(self.ob.get_objects_nearby, data.lat, data.long)
                # Кластеризация геометки
                cluster_future = executor.submit(self.clusterer.predict, data.lat, data.long)   


                ### РЕЗУЛЬТАТЫ ПАРАЛЛЕЛЬНЫХ ЗАДАЧ
                wd, td = weather_future.result()
                print(Fore.BLUE + '[!]' + Fore.RESET + f' Погодные данные: {wd.model_dump()}\n' +
                    Fore.BLUE + '[!]' + Fore.RESET + f' Временная метка: {td.model_dump()}')
                # Сбор прочих данных
                otd = other_model.OtherModel(light=1, speed=0, low_distance=0, op_way=0, give_way=0, village=0, city=1)
                print(Fore.BLUE + '[!]' + Fore.RESET + f' Прочие данные: {otd.model_dump()}')
                # Сбор данных о объектах поблизости
                od = objects_future.result()
                print(Fore.BLUE + '[!]' + Fore.RESET + f' Объекты поблизости: {od.model_dump()}')
                # Кластеризация геометки
                print(Fore.BLUE + '[!]' + Fore.RESET +' Кластеризация...')
                cluster = cluster_future.result()
                print(Fore.BLUE + '[!]' + Fore.RESET + f' Номер кластера: {cluster}')


                features = {'exp':[data.exp],'peop_in_car':[data.peop_in_car],'vehicle_age':[data.vehicle_age],'dayofweek_sin':[td.dayofweek_sin],
                'dayofweek_cos':[td.dayofweek_cos],'month_sin':[td.month_sin],'month_cos':[td.month_cos],'hour_sin':[td.hour_sin],
                'hour_cos':[td.hour_cos],'light':[otd.light],'gender':[data.gender],'rain':[wd.rain],'snow':[wd.snow],'cloudy':[wd.cloudy],'speed':[otd.speed],
                'low_distance':[otd.low_distance],'op_way':[otd.op_way],'give_way':[otd.give_way],'pedestrian_cross':[od.pedestrian_cross],'houses':[od.houses],'yard_exit':[od.yard_exit],
                'r_roadcross':[od.r_roadcross],'nr_roadcross':[od.nr_roadcross],'school':[od.school],'bridge':[od.bridge],'industry':[od.industry],'mall':[od.mall],'bus_stop':[od.bus_stop],
                'roundabout':[od.roundabout],'administration':[od.administration],'rwd':[data.rwd],'awd':[data.awd],'fwd':[data.fwd],'weight':[data.weight],'power':[data.power],
                'village':[otd.village],'city':[otd.city],'highway':[od.highway],'cluster':[cluster],'medium':[data.medium],'spacious':[data.spacious],
                'heavy': [data.heavy],'small': [data.small]}
                print(Fore.BLUE + '[!]' + Fore.RESET + f' Конечные признаки: {features}')
                
                ### НОРМАЛИЗАЦИЯ ПРИЗНАКОВ
                scaler = self.model.get_scaler()
                X = scaler.transform(pd.DataFrame(features))
                ### ПЕРЕВОД В СЕРОЕ ИЗОБРАЖЕНИЕ
                M_image = self.gia.convert(X, self.model)

                ### ПРЕДСКАЗАНИЕ
                tasp_cnn = self.model.get_model()
                result = tasp_cnn.predict(M_image)
                print(Fore.BLUE + '[!]' + Fore.RESET + f' Предсказание: {result}')
                predicted_class = np.argmax(result, axis=1)

                return {"severity": int(predicted_class[0])}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))