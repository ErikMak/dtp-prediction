from pydantic import BaseModel

class TimestampModel(BaseModel):
    dayofweek_sin: float
    dayofweek_cos: float
    month_sin: float
    month_cos: float
    hour_sin: float
    hour_cos: float