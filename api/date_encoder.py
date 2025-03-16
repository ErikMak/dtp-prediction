import datetime
import numpy as np
import pandas as pd
from pydantic_data.timestamp_model import TimestampModel

class DateEncoder:
    def encode(self, current_time=datetime.datetime.now()) -> TimestampModel:
        """
        Циклически кодирует дату через синусоидальные и косинусоидальные функции.
        
        :param current_time: Временная метка в формате datetime
        :return: Модель Pydantic TimestampModel
        """
        date = pd.to_datetime(current_time)

        timestamp = TimestampModel(
            dayofweek_sin=np.sin(2 * np.pi * date.day_of_week / 7),
            dayofweek_cos=np.cos(2 * np.pi * date.day_of_week / 7),
            month_sin=np.sin(2 * np.pi * date.month / 12),
            month_cos=np.cos(2 * np.pi * date.month / 12),
            hour_sin=np.sin(2 * np.pi * date.hour / 24),
            hour_cos=np.cos(2 * np.pi * date.hour / 24)
        )

        return timestamp