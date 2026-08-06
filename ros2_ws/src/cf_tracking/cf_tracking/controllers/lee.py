"""
Geometric SE(3) tracking controller(Lee, Leok, McClamroch 2010)

- Outer loop: We compute the desired translational acceleration from the pos/vel
  error and desired feedforward acceleration(PD controller with feedforward)
        a_cmd = a_ref + Kp (p_ref - p) + Kd (v_ref - v)
    - With gravity compensation we compute required thrust vector
        f_vec = m (a_cmd + g e3)
        thrust(compo in body z direction) = f_vec . (R e3)
    - we align the body's thrust axis with desired force while respecting the 
      reference yaw
        b3d = f_vec / |f_vec|, b1d from ref yaw, and with b2d we can construct Rd 

- Inner loop(geometric controller): We compute attitude and angular rate errors
  directly on SO(3) and then define controller with proportional, derivative, and 
  gyroscopic compensation
    - attitude error, eR = 0.5 (Rd^T R - R^T Rd)^vee
    - angular rate error, eW = omega - R^T Rd omega_d
    - Compute the control moments,
        M = -kR eR - kW eW + omega x (J omega)

- Now that we have total thrust as body moments, we get the motor speeds from the
  motor mixer
"""
import numpy as np

from cf_tracking.controller_base import Controller, ControlOutput

E3 = np.array([0.0, 0.0, 1.0]) # world z axis

def vee(m):
    return np.array([m[2, 1], m[0, 2], m[1, 0]])

class LeeController(Controller):
    name = 'lee'

    KP_POS = np.array([6.0, 6.0, 7.0]) # in 1/s^2
    KD_POS = np.array([4.0, 4.0, 4.5]) # in 1/s
    KR = np.array([1.0e-2, 1.0e-2, 3.3e-3]) # in N m / rad
    KW = np.array([7.7e-4, 7.7e-4, 5.1e-4]) # in N m s / rad
    A_XY_MAX = 2.5 # in m/s^2, horizontal accel clamp
    A_Z_MAX = 3.0 # in m/s^2, vertical accel clamp

    def gains(self):
        return {'kp_pos': self.KP_POS.tolist(), 'kd_pos': self.KD_POS.tolist(),
                'kR': self.KR.tolist(), 'kW': self.KW.tolist(),
                'a_xy_max': self.A_XY_MAX, 'a_z_max': self.A_Z_MAX}

    def compute(self, t, ref, state):
        '''
        Called every control cycle(here at 250Hz)
        '''
        a_cmd = (ref.acc + self.KP_POS * (ref.pos - state.pos)
                 + self.KD_POS * (ref.vel - state.vel))
        return self.accel_to_output(a_cmd, ref, state)

    def accel_to_output(self, a_cmd, ref, state):
        '''
        Shared(common for all controllers) inner loop pipeline: 
        From the a_cmd we get the desired thrust, attitude, body moment and 
        finally the resulting motor speeds
        '''
        v = self.vehicle
        a_cmd = np.array(a_cmd, float)
        
        n_xy = np.hypot(a_cmd[0], a_cmd[1])
        if n_xy > self.A_XY_MAX:
            a_cmd[:2] *= self.A_XY_MAX / n_xy
        a_cmd[2] = np.clip(a_cmd[2], -self.A_Z_MAX, self.A_Z_MAX)

        f_vec = v.mass * (a_cmd + v.gravity * E3)
        f_norm = np.linalg.norm(f_vec)
        if f_norm < 0.1 * v.hover_thrust:
            f_vec = 0.1 * v.hover_thrust * E3
            f_norm = 0.1 * v.hover_thrust
        thrust = float(f_vec @ (state.R @ E3))
        thrust = max(thrust, 0.0)

        b3d = f_vec / f_norm
        b1c = np.array([np.cos(ref.yaw), np.sin(ref.yaw), 0.0])
        b2d = np.cross(b3d, b1c)
        b2d /= np.linalg.norm(b2d)
        b1d = np.cross(b2d, b3d)
        Rd = np.column_stack([b1d, b2d, b3d])

        eR = 0.5 * vee(Rd.T @ state.R - state.R.T @ Rd)
        omega_d = Rd.T @ np.array([0.0, 0.0, ref.yaw_rate])
        eW = state.omega - state.R.T @ Rd @ omega_d
        M = (-self.KR * eR - self.KW * eW
             + np.cross(state.omega, v.inertia * state.omega))

        w, saturated = self.mixer.to_motor_speeds(thrust, M)
        att_err = np.degrees(np.arccos(np.clip(b3d @ (state.R @ E3), -1, 1)))
        return ControlOutput(motor_speeds=w, thrust=thrust, moments=M,
                             saturated=saturated, att_err_deg=float(att_err))