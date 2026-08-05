"""
To test multiple controllers(Lee, Millinger, MPC) under same conditions, we
define a abstract Controller class(with abstract method compute() and other 
non abstract methods). Helps to swape any controller, all takes same 
inputs(t, ref, state) and returns same type controlOutput
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

@dataclass
class ControlOutput:
    motor_speeds: np.ndarray # (4,) rad/s, actuator command sent to gz
    thrust: float = float('nan')
    moments: np.ndarray = field(default_factory=lambda: np.full(3, np.nan))
    saturated: bool = False
    att_err_deg: float = float('nan') # angle between body z and desired 

class Controller(ABC):
    '''
    The controllers name and gains will be written into a metadata 
    file(meta.json), so we will know which controller and tuning 
    parameters were used for a sim run

    Predictive controller like MPC needs references at future time 
    instances, so it will use the ref = self.preview(t) function
    '''
    name = 'base'

    def __init__(self, vehicle, mixer):
        self.vehicle = vehicle
        self.mixer = mixer
        self.preview = None

    def gains(self):
        '''
        Returns control gains for logging, default to empty dictionary
        '''
        return {}

    def reset(self):
        '''
        Some controller needs to reset there internal states like integral 
        error in PID or optimizer warm start
        '''
        pass

    @abstractmethod
    def compute(self, t, ref, state) -> ControlOutput:
        """t: sim time [s]; ref: references.Reference; state: state.State."""
        '''
        Mandatory method that every controller must implement

        t : Current sim time
        ref : Desired reference(target position, velocity, orientation)
        state : Current measured or estimated state of the vehicle

        reture controlOutput object
        '''