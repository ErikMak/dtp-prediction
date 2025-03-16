import openmeteo_requests
from date_encoder import DateEncoder
import requests_cache
from retry_requests import retry
from pydantic_data.weather_model import WeatherModel
from pydantic_data.timestamp_model import TimestampModel
import datetime

class Weather:
    openmeteo = None
    _URL = "https://api.open-meteo.com/v1/forecast"
    _date_encoder = None

    def __init__(self):
        cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
        retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
        self.openmeteo = openmeteo_requests.Client(session = retry_session)
        self._date_encoder = DateEncoder()

    def _convert_time(self, unixtime: int) -> TimestampModel:
        return self._date_encoder.encode(datetime.datetime.fromtimestamp(unixtime))


    def get_current_weather(self, lat: float, long: float) -> tuple[WeatherModel, TimestampModel]:
        """
        Возвращает кодированную текущую погоду и время с погодного API по координатам.
        
        :param lat: Широта - число с плавающей точкой
        :param long: Долгота - число с плавающей точкой
        :return: Кортеж моделей Pydantic WeatherModel и Pydantic TimestampModel
        """
        params = {
            "latitude": lat,
            "longitude": long,
            "current": ["weather_code", "rain", "snowfall"],
            "timeformat": "unixtime"
        }

        responses = self.openmeteo.weather_api(self._URL, params=params)

        response = responses[0]
        current = response.Current()
        current_weather_code = current.Variables(0).Value()
        current_rain = current.Variables(1).Value()
        current_snowfall = current.Variables(2).Value()

        cloudy, rain, snow = 0, 0, 0
        codes = [3, 51, 53, 61, 63, 65, 55]
        if current_weather_code in codes:
            cloudy = 1
        if current_rain >= 4:
            rain = 1
        if current_snowfall >= 7:
            snow = 1

        timestamp = self._convert_time(current.Time())
        weather = WeatherModel(cloudy=cloudy, rain=rain, snow=snow)

        return weather, timestamp