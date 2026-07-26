"""
Vehicle state container assembled from odometry + IMU
"""

import math
from dataclasses import dataclass

import numpy as np


def quat_to_rot(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def quat_to_yaw_tilt(x, y, z, w):
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (x * x + y * y)))))
    return yaw, tilt

'''
Stores all the sates in a State object
'''
@dataclass
class State:
    pos: np.ndarray # (3,) in world frame
    vel: np.ndarray # (3,) in world frame
    R: np.ndarray # (3,3) body to world
    quat: np.ndarray # (4,) x y z w
    omega: np.ndarray # (3,) body rates(from IMU)
    yaw: float
    tilt_deg: float

    @classmethod
    def from_msgs(cls, odom, omega_body):
        p = odom.pose.pose.position
        q = odom.pose.pose.orientation
        v = odom.twist.twist.linear # in body frame(odom child frame)
        R = quat_to_rot(q.x, q.y, q.z, q.w)
        yaw, tilt = quat_to_yaw_tilt(q.x, q.y, q.z, q.w)
        return cls(pos=np.array([p.x, p.y, p.z]),
                   vel=R @ np.array([v.x, v.y, v.z]),
                   R=R, quat=np.array([q.x, q.y, q.z, q.w]),
                   omega=np.asarray(omega_body, float),
                   yaw=yaw, tilt_deg=tilt)