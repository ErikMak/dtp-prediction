from pydantic import BaseModel

class Data(BaseModel):
    long: float
    lat: float
    peop_in_car: int
    vehicle_age: int
    exp: int
    gender: int
    rwd: bool
    awd: bool
    fwd: bool
    weight: int
    power: int
    medium: bool
    spacious: bool
    heavy: bool
    small: bool