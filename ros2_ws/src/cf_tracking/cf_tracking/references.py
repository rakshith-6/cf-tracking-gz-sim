"""
Adds higher order derivatives acc(t) and jerk(t) and packs everything into 
a single Reference object that the tracking controller uses through sample(t)

The returned reference trajs are expressed in terms of the flat outputs {x, y, z, psi} 
and there derivatives, making them helpful for the flatness based tracking controllers
"""
from dataclasses import dataclass, field

import numpy as np

from cf_tracking.trajectory_lib import build_trajectory

_EPS = 1e-3   # in s, half-step for acc

@dataclass
class Reference:
    pos: np.ndarray
    vel: np.ndarray # (3,) world
    acc: np.ndarray # (3,) world
    yaw: float
    yaw_rate: float
    jerk: np.ndarray = field(default_factory=lambda: np.zeros(3)) # (3,) world

class RefTrajectory:
    def __init__(self, traj):
        self.traj = traj
        self.duration = traj.duration

    def acc(self, t):
        t0 = max(0.0, t - _EPS)
        t1 = min(self.duration, t + _EPS)
        if t1 <= t0:
            return np.zeros(3)
        return (self.traj.vel(t1) - self.traj.vel(t0)) / (t1 - t0)

    def jerk(self, t):
        t0 = max(0.0, t - _EPS)
        t1 = min(self.duration, t + _EPS)
        if t1 <= t0:
            return np.zeros(3)
        return (self.acc(t1) - self.acc(t0)) / (t1 - t0)

    def sample(self, t):
        return Reference(pos=self.traj.pos(t), vel=self.traj.vel(t),
                         acc=self.acc(t), yaw=self.traj.yaw(t),
                         yaw_rate=self.traj.yaw_rate(t), jerk=self.jerk(t))

def make_setpoint(pos, yaw=0.0):
    '''
    Constant hover reference:
    For hover tests, takeoff/landing holds, pause after tracking a trajectory
    '''
    p = np.asarray(pos, float)
    return Reference(pos=p, vel=np.zeros(3), acc=np.zeros(3),
                     yaw=float(yaw), yaw_rate=0.0)

def build_reference(primitive, **kwargs):
    return RefTrajectory(build_trajectory(primitive, **kwargs))