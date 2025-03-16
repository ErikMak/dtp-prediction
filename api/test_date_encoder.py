from date_encoder import DateEncoder
import datetime

def test_encode():
    d_e = DateEncoder()

    date = d_e.encode(datetime.datetime(2020, 10, 5,14, 22, 56))
    date = date.model_dump()

    assert date['dayofweek_sin'] == 0.0
    assert date['dayofweek_cos'] == 1.0
    assert date['month_sin'] == -0.8660254037844386
    assert date['month_cos'] == 0.5000000000000001
    assert date['hour_sin'] == -0.4999999999999997
    assert date['hour_cos'] == -0.8660254037844388