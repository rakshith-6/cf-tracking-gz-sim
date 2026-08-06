"""
Controller registry :
All the new controllers are added here
"""
from cf_tracking.controllers.lee import LeeController

CONTROLLERS = {
    'lee': LeeController,
}

def make_controller(name, vehicle, mixer):
    try:
        return CONTROLLERS[name](vehicle, mixer)
    except KeyError:
        raise ValueError(
            f'unknown controller {name!r}; available: {sorted(CONTROLLERS)}')