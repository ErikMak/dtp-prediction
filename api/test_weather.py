from request.weather import Weather

def test_convert_time():
    unix = 1601896920
    w = Weather()

    date = w._convert_time(unix)
    date = date.model_dump()

    assert date['dayofweek_sin'] == 0.0
    assert date['dayofweek_cos'] == 1.0
    assert date['month_sin'] == -0.8660254037844386
    assert date['month_cos'] == 0.5000000000000001
    assert date['hour_sin'] == -0.4999999999999997
    assert date['hour_cos'] == -0.8660254037844388

def test_get_current_weather():
    long = 31.295275
    lat = 58.535864

    w = Weather()
    wd, td = w.get_current_weather(lat, long)

    wd = wd.model_dump()
    assert len(wd) == 3
    assert wd["cloudy"] == 1 or wd["cloudy"] == 0