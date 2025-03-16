from pydantic import BaseModel

class OtherModel(BaseModel):
    light: bool
    speed: bool
    low_distance: bool
    op_way: bool
    give_way: bool
    village: bool
    city: bool