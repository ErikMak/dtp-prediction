from pydantic import BaseModel

class ObjectsModel(BaseModel):
    pedestrian_cross: bool
    houses: bool
    yard_exit: bool
    r_roadcross: bool
    nr_roadcross: bool
    school: bool
    bridge: bool
    industry: bool
    mall: bool
    bus_stop: bool
    roundabout: bool
    administration: bool
    highway: bool