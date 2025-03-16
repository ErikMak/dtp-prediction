from pydantic import BaseModel
from pydantic_data.timestamp_model import TimestampModel
from pydantic_data.objects_model import ObjectsModel
from pydantic_data.weather_model import WeatherModel
from pydantic_data.other_model import OtherModel

class FeaturesModel(BaseModel):
    exp: int
    peop_in_car: int
    vegicle_age: int
    timestampModel: TimestampModel
    gender: bool
    weatherModel:WeatherModel
    otherModel: OtherModel
    objectsModel: ObjectsModel
    rwd: bool
    awd: bool
    fwd: bool
    weight: int
    power: int
    cluster: int
    medium: bool
    spacious: bool
    heavy: bool
    small: bool