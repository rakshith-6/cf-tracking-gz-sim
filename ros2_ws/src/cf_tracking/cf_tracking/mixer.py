"""
X-configuration mixer: 

Motor table(FLU body frame, matches sim/models/crazyflie_tracking/model.sdf)
    m1(+,-)-ccw:front-right
    m2(-,-)-cw:back-right
    m3(-,+)-ccw:back-left
    m4(+,+)-cw:front-left

Forward map, with f_i = C_T w_i^2, m_i = C_Q w_i^2, a = arm length,
    T = f1 + f2 + f3 + f4
    tau_x = a (-f1 - f2 + f3 + f4)
    tau_y = a (-f1 + f2 + f3 - f4)
    tau_z = -m1 + m2 - m3 + m4

The 4x4 allocation matrix has mutually orthogonal rows, so the inverse is
its transpose scaled by 1/4,
M.(1/4 M^T) = I
"""
import numpy as np

class Mixer:
    def __init__(self, vehicle):
        self.v = vehicle

    def to_motor_speeds(self, thrust, moments):
        '''
        Inverse map : From controller output of (thrust, moments) 
        we get 4 rotor speeds

        - Negative per rotor thrust is clipped to zero, speed is clamped to 
          [motor_speed_min, motor_speed_max]
        - If `staturated` is retured true(more speed than what is possible) 
          we will know that the clipping happened 
        '''

        a, kappa = self.v.arm, self.v.moment_constant
        b0 = thrust
        b1 = moments[0] / a
        b2 = moments[1] / a
        b3 = moments[2] / kappa
        f = 0.25 * np.array([b0 - b1 - b2 - b3,
                             b0 - b1 + b2 + b3,
                             b0 + b1 + b2 - b3,
                             b0 + b1 - b2 + b3])
        saturated = bool((f < 0.0).any())
        w = np.sqrt(np.maximum(f, 0.0) / self.v.c_t)
        w_clamped = np.clip(w, self.v.motor_speed_min, self.v.motor_speed_max)
        saturated = saturated or bool((w != w_clamped).any())
        return w_clamped, saturated

    def to_wrench(self, w):
        '''
        Forward map: from motor speeds to thrust and torques, for logging 
        and tests
        '''
        w2 = np.asarray(w, float) ** 2
        f = self.v.c_t * w2
        m = self.v.c_q * w2
        a = self.v.arm
        return float(f.sum()), np.array([
            a * (-f[0] - f[1] + f[2] + f[3]),
            a * (-f[0] + f[1] + f[2] - f[3]),
            -m[0] + m[1] - m[2] + m[3]])