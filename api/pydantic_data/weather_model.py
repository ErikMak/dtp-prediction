from pydantic import BaseModel

class WeatherModel(BaseModel):
    cloudy: bool
    rain: bool
    snow: bool