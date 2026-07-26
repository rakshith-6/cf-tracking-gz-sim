"""
Crazyflie parameter are stored in an object called Vehicle and loaded 
from config/vehicle.yaml
"""

import os
from dataclasses import dataclass, field

import numpy as np
import yaml

@dataclass
class Vehicle:
    mass: float
    gravity: float
    inertia: np.ndarray # (3,) diagonal elements
    c_t: float
    moment_constant: float
    arm: float
    motor_speed_max: float
    motor_speed_min: float = 0.0

    @property
    def c_q(self):
        return self.moment_constant * self.c_t

    @property
    def hover_thrust(self):
        return self.mass * self.gravity

    @property
    def hover_motor_speed(self):
        return float(np.sqrt(self.hover_thrust / (4.0 * self.c_t)))


def load_vehicle(path=None):
    '''
    Loads parameters from the vehicle.yaml file
    '''
    if path is None:
        from ament_index_python.packages import get_package_share_directory
        path = os.path.join(get_package_share_directory('cf_tracking'),
                            'config', 'vehicle.yaml')
    with open(path) as f:
        d = yaml.safe_load(f)
    return Vehicle(
        mass=float(d['mass']),
        gravity=float(d['gravity']),
        inertia=np.array([d['inertia']['ixx'], d['inertia']['iyy'],
                          d['inertia']['izz']], float),
        c_t=float(d['c_t']),
        moment_constant=float(d['moment_constant']),
        arm=float(d['arm']),
        motor_speed_max=float(d['motor_speed_max']),
        motor_speed_min=float(d.get('motor_speed_min', 0.0)))