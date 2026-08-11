"""
Script to execute the entire flight experiment and logging

Modes:
1. track: takeoff, track one of trajectory_lib primitive and, break
2. hover: hold (0, 0, altitude) for a specified duration(to tuning gains, 
   verify stability and to test disturbance rejection)
3. open_loop: constant motor speed for some specified duration(no controller, 
   no takeoff, a slow climb at ~1.02x hover speed) for smoke testing the 
   pipeline

- log.csv: Log of everything measured at every timestep - time, pos, vel, 
  errors, motor speeds, thrust, moments. To be used for plotting(250Hz, 
  so 250 log entries every sec)
- meta.json : We save the experiment config here (controller, traj, vehicle 
  mass, wind, gains)
- metrics.json: Perfoemace metrics like max error and saturation 
  percentage(scripts/plot_trajectory.py). Exit code retured 0 if success 
  and 2 if aborted
"""
import json
import math
import os
import sys
import time
from datetime import datetime

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from actuator_msgs.msg import Actuators # for motor ang vel
from nav_msgs.msg import Odometry # for pose and linear vel
from sensor_msgs.msg import Imu # for ang vel and line accl(not used)

from cf_tracking.controller_base import ControlOutput
from cf_tracking.controllers import make_controller
from cf_tracking.mixer import Mixer
from cf_tracking.references import Reference, build_reference, make_setpoint
from cf_tracking.state import State
from cf_tracking.vehicle import load_vehicle

TILT_ABORT_DEG = 60.0
BOX_Z = 4.0 # max abort altitude
LOG_FLUSH_EVERY = 100 # rows buffered and pushed to csv file 

LOG_HEADER = (
    't,phase,'
    'ref_px,ref_py,ref_pz,ref_vx,ref_vy,ref_vz,ref_yaw,'
    'px,py,pz,vx,vy,vz,qx,qy,qz,qw,wx,wy,wz,yaw,tilt_deg,'
    'err_px,err_py,err_pz,err_pos,err_vx,err_vy,err_vz,err_vel,'
    'err_yaw,att_err_deg,'
    'thrust,mx,my,mz,w1,w2,w3,w4,saturated,dt_sim,odom_age')

def _min_jerk(p0, p1, T, t):
    '''
    Given start, destination points and time step, the function solves for a 
    minimum jerk trajectory point(pos, vel, acc at t in [0, T])
    '''
    s = np.clip(t / T, 0.0, 1.0)
    b = 10 * s**3 - 15 * s**4 + 6 * s**5
    db = (30 * s**2 - 60 * s**3 + 30 * s**4) / T
    ddb = (60 * s - 180 * s**2 + 120 * s**3) / T**2
    d = p1 - p0
    return p0 + b * d, db * d, ddb * d

class HarnessFlight(Node):
    def __init__(self):
        super().__init__('harness_flight')
        self.set_parameters([rclpy.parameter.Parameter(
            'use_sim_time', rclpy.Parameter.Type.BOOL, True)])
        '''
        I have declared some parameters with default values, some of other
        available options and more details are as follows,

        controller: lee, millinger, mpc
        mode: track, hover, open_loop
        primitive: circle, harmonic, rose, lissajous, spline, superellipse
        yaw_mode: fixed, sinusoidal, velocity, rate_const
        seed: We will use same seed when running any primitive on all controllers
        max_speed: limits the peak speed of any trajectory
        duration: nominal experiment duration(time scaling can increase this)
        altitude: reference flight altitude
        radius: for circle only
        box_xy: allowed flight workspace half width, prevents runaway controllers
        cmd_rate: controller excecutes every 0.004s or at 250Hz as default
        out_dir: every experiment creates log.csv, meta.json and metrics.json in 
                 this directory run_name/data/tracking_runs 
        run_name: harness automatically generates folder name, we can override it 
                  by passing name with parameter
        open_loop_speed_scale: thrust for gentle climb, for pipeline test purpose
        wind_x, wind_y and wind_z: wind velocity in world coordinates
        wind_onset_delay: -1 means the wind exists from the beginning or s seconds
                          after entering primitive/hover phase(no wind in takeoff 
                          phase) wind is turned on via the runtime gz wind topic
        mass_scale and inertia_scale: domain randomization parameters, increasing
                                      the mass and inertia values for better sim to 
                                      real transition(recorded in meta.json)

        To actually use the parameters values, we can call get_parameter() to 
        return a Parameter object and assign its value to a variable using .value
        '''
        self.declare_parameter('controller', 'lee')
        self.declare_parameter('mode', 'track')
        self.declare_parameter('primitive', 'circle')
        self.declare_parameter('yaw_mode', 'fixed')
        self.declare_parameter('seed', 0)
        self.declare_parameter('max_speed', 0.5)
        self.declare_parameter('duration', 30.0)
        self.declare_parameter('altitude', 1.0)
        self.declare_parameter('radius', 0.6)
        self.declare_parameter('box_xy', 3.0)
        self.declare_parameter('cmd_rate', 250.0)
        self.declare_parameter('out_dir', '/data/tracking_runs')
        self.declare_parameter('run_name', '')
        self.declare_parameter('open_loop_speed_scale', 1.02)
        self.declare_parameter('wind_x', 0.0)
        self.declare_parameter('wind_y', 0.0)
        self.declare_parameter('wind_z', 0.0)
        self.declare_parameter('wind_onset_delay', -1.0)
        self.declare_parameter('mass_scale', 1.0)
        self.declare_parameter('inertia_scale', 1.0)

        self.mode = self.get_parameter('mode').value
        self.ctrl_name = self.get_parameter('controller').value
        self.altitude = float(self.get_parameter('altitude').value)
        self.duration = float(self.get_parameter('duration').value)
        self.cmd_rate = float(self.get_parameter('cmd_rate').value)
        self.box_xy = float(self.get_parameter('box_xy').value)

        self.vehicle = load_vehicle()
        self.mixer = Mixer(self.vehicle)
        self.controller = None
        self.ref_traj = None
        
        if self.mode == 'track':
            self.ref_traj = build_reference(
                self.get_parameter('primitive').value,
                yaw_mode=self.get_parameter('yaw_mode').value,
                seed=self.get_parameter('seed').value,
                max_speed=float(self.get_parameter('max_speed').value),
                duration=self.duration,
                altitude=self.altitude,
                radius=float(self.get_parameter('radius').value))
        
        if self.mode != 'open_loop':
            self.controller = make_controller(self.ctrl_name, 
                                              self.vehicle, 
                                              self.mixer)

        # We will create a run directory + logging
        label = (self.get_parameter('primitive').value
                 if self.mode == 'track' else self.mode)
        run_name = self.get_parameter('run_name').value or (
            f"{datetime.now():%Y%m%d_%H%M%S}_"
            f"{self.ctrl_name if self.controller else 'none'}_{label}")
        self.run_dir = os.path.join(self.get_parameter('out_dir').value,
                                    run_name)
        os.makedirs(self.run_dir, exist_ok=True)
        
        self.log_f = open(os.path.join(self.run_dir, 'log.csv'), 'w')
        self.log_f.write(LOG_HEADER + '\n')
        self.log_rows = 0
        self._write_meta()
        self.get_logger().info(f'run dir: {self.run_dir}')

        '''
        metric accumulators(for primitive/hover phase only)

        m_n: no. of samples contibuting to the metric  
        m_pos2: cumulative position squared error(will be used for RMSE)
        m_vel2: velocity squared error accumulator
        m_yaw2: yaw squared error accumulator
        m_pos_max: maximum position error in the run
        m_sat: counts how many controller outputs caused motor saturation
        wall_t0 and sim_t0: wall clock(real computer time) and gazebo 
                            simulation start times(used for RTF cal)
        '''
        self.m_n = 0
        self.m_pos2 = np.zeros(3)
        self.m_vel2 = 0.0
        self.m_yaw2 = 0.0
        self.m_pos_max = 0.0
        self.m_sat = 0
        self.wall_t0 = self.sim_t0 = None

        '''
        - Publisher: Angular velocity commands from the controller with 
                     message type actuator_msgs/msg/Actuators is 
                     published at the topic '/crazyflie/motor_speed'. 
                     The last arguments is the number of messages the 
                     publisher can buffer(here it is upto 10)

        - QoS profile for subscribers: We are using depth=1 to get newest 
                                       messages. We skip the state lost 
                                       and only recieve newest ones(best 
                                       effort)

        - Subscribers: Whenever the respective messages arrive, it subcribes 
                       to crazyflie's odometery and IMU topic 

        - The controller wont command motors until the vaild sensor data 
          ,odometery and IMU data arrives(until then in wait_state phase). 
        '''
        self.cmd_pub = self.create_publisher(
                                Actuators, '/crazyflie/motor_speed', 10)
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Odometry, '/crazyflie/odometry', self.on_odom, qos)
        self.create_subscription(Imu, '/crazyflie/imu', self.on_imu, qos)

        self.odom = None
        self.omega = np.zeros(3)
        self.have_imu = False
        self.last_t = None
        self.phase = 'wait_state'
        self.phase_start = None # phase start time
        self.takeoff_from = None # takeoff position
        self.takeoff_T = None # takeoff duration
        self.brake_ref = None
        self.abort = False

        self.wind_onset_delay = float(self.get_parameter('wind_onset_delay').value)
        self._wind_fired = False
        self.timer = self.create_timer(1.0 / self.cmd_rate, self.step)

    '''
    ROS2 subscribers callbacks:

    - Whenever a new odomentery message arries on /crazyflie/odometry, call 
      on_odom(msg) and stores latest measurment. The msg argument contains 
      the complete odometery message
    - Everytime an IMU message arrives, ROS calls on_IMU(msg) and we extract
      angular velocity
    - now_s() returns current ROS time in seconds

    Logging and bookkeep funs: 
    
    _write_meta() writes the meta.json with all the experiment details and 
    parmeters. _log() estimates errors, updates performance metrics(cumulative
    squared error, max position error, % saturation) and records along with 
    observations every control timestep. _write_metrics() calculates useful 
    metrics(like RMSEs) which summerizes the experiment(writes to metrics.json)
    '''
    def on_odom(self, msg):
        self.odom = msg

    def on_imu(self, msg):
        w = msg.angular_velocity
        self.omega = np.array([w.x, w.y, w.z])
        self.have_imu = True

    def now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _write_meta(self):
        meta = {
            'mode': self.mode,
            'controller': self.ctrl_name if self.controller else None,
            'gains': self.controller.gains() if self.controller else {},
            'primitive': self.get_parameter('primitive').value,
            'yaw_mode': self.get_parameter('yaw_mode').value,
            'seed': self.get_parameter('seed').value,
            'max_speed': float(self.get_parameter('max_speed').value),
            'duration_param': self.duration,
            'duration_scaled': (self.ref_traj.duration if self.ref_traj
                                else self.duration),
            'altitude': self.altitude,
            'cmd_rate': self.cmd_rate,
            'wind': [float(self.get_parameter(p).value)
                     for p in ('wind_x', 'wind_y', 'wind_z')],
            'mass_scale': float(self.get_parameter('mass_scale').value),
            'inertia_scale': float(self.get_parameter('inertia_scale').value),
            'vehicle': {'mass': self.vehicle.mass,
                        'inertia': self.vehicle.inertia.tolist(),
                        'c_t': self.vehicle.c_t,
                        'moment_constant': self.vehicle.moment_constant,
                        'arm': self.vehicle.arm,
                        'motor_speed_max': self.vehicle.motor_speed_max},
            'started_wall': datetime.now().isoformat(timespec='seconds'),
        }
        with open(os.path.join(self.run_dir, 'meta.json'), 'w') as f:
            json.dump(meta, f, indent=2)

    def _log(self, t, ref, st, out, dt_sim):
        err_p = ref.pos - st.pos
        err_v = ref.vel - st.vel
        err_yaw = (ref.yaw - st.yaw + math.pi) % (2 * math.pi) - math.pi
        att_err = getattr(out, 'att_err_deg', float('nan'))
        row = [t, self.phase,
               *ref.pos, *ref.vel, ref.yaw,
               *st.pos, *st.vel, *st.quat, *st.omega, st.yaw, st.tilt_deg,
               *err_p, np.linalg.norm(err_p), *err_v, np.linalg.norm(err_v),
               err_yaw, att_err,
               out.thrust, *out.moments, *out.motor_speeds,
               int(out.saturated), dt_sim,
               t - (self.odom.header.stamp.sec
                    + self.odom.header.stamp.nanosec * 1e-9)]
        self.log_f.write(','.join(
            x if isinstance(x, str) else f'{x:.6g}' for x in row) + '\n')
        self.log_rows += 1
        if self.log_rows % LOG_FLUSH_EVERY == 0:
            self.log_f.flush()
        if self.phase in ('primitive', 'hover'):
            self.m_n += 1
            self.m_pos2 += err_p**2
            self.m_vel2 += float(err_v @ err_v)
            self.m_yaw2 += err_yaw**2
            self.m_pos_max = max(self.m_pos_max, float(np.linalg.norm(err_p)))
            self.m_sat += int(out.saturated)

    def _write_metrics(self):
        metrics = {'status': 'abort' if self.abort else 'ok',
                   'samples': self.m_n}
        if self.m_n:
            rmse_ax = np.sqrt(self.m_pos2 / self.m_n)
            metrics.update({
                'pos_rmse': float(np.sqrt(self.m_pos2.sum() / self.m_n)),
                'pos_rmse_xyz': rmse_ax.tolist(),
                'pos_err_max': self.m_pos_max,
                'vel_rmse': float(np.sqrt(self.m_vel2 / self.m_n)),
                'yaw_rmse': float(np.sqrt(self.m_yaw2 / self.m_n)),
                'saturated_frac': self.m_sat / self.m_n})
        if self.wall_t0 is not None:
            wall = time.monotonic() - self.wall_t0
            sim = self.now_s() - self.sim_t0
            if wall > 0:
                metrics['mean_rtf'] = sim / wall
        with open(os.path.join(self.run_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)

    '''
    Set of functions to set required phase, record the phase change time
    reset controller. Check for saftey(excessive tilt, ground contact and
    workspace boundary) and abort flight if not safe.  
    '''
    def set_phase(self, phase):
        pos = None
        if self.odom is not None:
            p = self.odom.pose.pose.position
            pos = np.round([p.x, p.y, p.z], 2)
        self.get_logger().info(
            f'phase: {self.phase} -> {phase} (t={self.now_s():.2f}, pos={pos})')
        self.phase = phase
        self.phase_start = self.now_s()
        if self.controller:
            self.controller.reset()

    def safety_ok(self, st):
        if st.tilt_deg > TILT_ABORT_DEG:
            self.get_logger().error(f'ABORT: tilt {st.tilt_deg:.0f} deg')
        elif self.phase not in ('wait_state', 'open_loop') and st.pos[2] < 0.02 \
                and self.now_s() - self.phase_start > 2.0:
            self.get_logger().error('ABORT: ground contact')
        elif abs(st.pos[0]) > self.box_xy or abs(st.pos[1]) > self.box_xy \
                or st.pos[2] > BOX_Z:
            self.get_logger().error(f'ABORT: out of box {np.round(st.pos, 2)}')
        else:
            return True
        return False

    '''
    _publish(): fun to publish the motor commands to crazyflie
    _finish(): fun to shutdown experiment and write metrics.json
    step(): starting form wait_state we wait till we get the odom 
            and IMU data(sim_time>0) for track and hover mode(not 
            for open_loop).
    '''
    def _publish(self, w):
        msg = Actuators()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.velocity = [float(x) for x in w]
        self.cmd_pub.publish(msg)

    def _finish(self, abort):
        self.abort = abort
        self._write_metrics()
        self.log_f.flush()
        self.log_f.close()
        self._publish(np.zeros(4))
        self.timer.cancel()
        raise SystemExit
 
    '''
    Read current state -> Calculate simulation time -> Safety check -> Determine 
    current flight phase -> Generate reference -> Controller computes motor 
    commands via mixer -> Log everything and publish motor speeds

    ref_traj: entire reference trajectory
    ref: reference at one particular instant/current time
    st: actual measured/estimated vehicle state
    '''
    def step(self):
        if self.phase == 'wait_state':
            need_imu = self.mode != 'open_loop'
            if self.odom is not None and (self.have_imu or not need_imu) \
                    and self.now_s() > 0.0:
                self.wall_t0 = time.monotonic()
                self.sim_t0 = self.now_s()
                if self.mode == 'open_loop':
                    self.set_phase('open_loop')
                else:
                    st = State.from_msgs(self.odom, self.omega)
                    self.takeoff_from = st.pos.copy()
                    goal = (self.ref_traj.sample(0.0).pos if self.ref_traj
                            else np.array([0.0, 0.0, self.altitude]))
                    dist = float(np.linalg.norm(goal - self.takeoff_from))
                    self.takeoff_T = max(3.0, dist / 0.3)
                    self.set_phase('takeoff')
            return

        st = State.from_msgs(self.odom, self.omega)
        t = self.now_s()
        dt_sim = 0.0 if self.last_t is None else t - self.last_t
        self.last_t = t
        t_phase = t - self.phase_start

        if not self.safety_ok(st):
            self._finish(abort=True)

        if self.phase == 'open_loop':
            w_cmd = self.vehicle.hover_motor_speed * float(
                self.get_parameter('open_loop_speed_scale').value)
            out_w = np.full(4, w_cmd)
            thrust, moments = self.mixer.to_wrench(out_w)
            out = ControlOutput(motor_speeds=out_w, thrust=thrust,
                                moments=moments)
            ref = make_setpoint(st.pos)
            self._log(t, ref, st, out, dt_sim)
            self._publish(out_w)
            if t_phase >= self.duration:
                self.get_logger().info('open_loop complete')
                self._finish(abort=False)
            return

        if self.phase == 'takeoff':
            goal = (self.ref_traj.sample(0.0) if self.ref_traj
                    else make_setpoint([0.0, 0.0, self.altitude]))
            p, v, a = _min_jerk(self.takeoff_from, goal.pos,
                                self.takeoff_T, t_phase)
            ref = Reference(pos=p, vel=v, acc=a, yaw=goal.yaw, yaw_rate=0.0)
            err = float(np.linalg.norm(st.pos - goal.pos))
            '''
            We give extra 1s to settle at the goal pos after takeoff and proceed
            if error is within 15cm. We give another extra 20s and error margin
            of 40cm with a printed message(as no integral term to counter wind)
            '''
            if t_phase > self.takeoff_T + 1.0 and err < 0.15:
                self.set_phase('hover' if self.mode == 'hover'
                               else 'primitive')
            elif t_phase > self.takeoff_T + 20.0:
                if err < 0.4:
                    self.get_logger().warning(
                        f'takeoff converged to {err:.2f} m offset (wind?); '
                        'proceeding')
                    self.set_phase('hover' if self.mode == 'hover'
                                   else 'primitive')
                else:
                    self.get_logger().error('ABORT: takeoff timeout')
                    self._finish(abort=True)

        elif self.phase == 'hover':
            ref = make_setpoint([0.0, 0.0, self.altitude])
            if t_phase >= self.duration:
                self.brake_ref = ref
                self.set_phase('brake')

        elif self.phase == 'primitive':
            ref = self.ref_traj.sample(t_phase)
            if t_phase >= self.ref_traj.duration:
                end = self.ref_traj.sample(self.ref_traj.duration)
                self.brake_ref = make_setpoint(end.pos, end.yaw)
                self.set_phase('brake')

        elif self.phase == 'brake':
            ref = self.brake_ref
            if t_phase >= 2.0:
                self.get_logger().info('tracking run complete')
                self._finish(abort=False)
        '''
        We command wind mid flight after wind onset delay time(only during 
        experimental phase), _wind_fired ensures we implement this once. 
        Then we send command to Gazebo by using subprocess module to launch
        a process to excecute gz topics
        '''
        if (self.wind_onset_delay >= 0.0 and not self._wind_fired
                and self.phase in ('primitive', 'hover')
                and t_phase >= self.wind_onset_delay):
            self._wind_fired = True
            wx, wy, wz = (float(self.get_parameter(p).value)
                          for p in ('wind_x', 'wind_y', 'wind_z'))
            import subprocess
            subprocess.Popen(
                ['gz', 'topic', '-t', '/world/cf_tracking/wind',
                 '-m', 'gz.msgs.Wind', '-p',
                 f'linear_velocity {{x: {wx}, y: {wy}, z: {wz}}}, '
                 f'enable_wind: true'],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.get_logger().info(
                f'WIND ONSET at t_phase={t_phase:.2f}s: '
                f'({wx}, {wy}, {wz}) m/s')

        '''
        We give controller.preview(t) access to the reference trajectory
        function(for predictive controllers like MPC) 
        '''
        self.controller.preview = (
            self.ref_traj.sample if self.phase == 'primitive' else None)
        
        out = self.controller.compute(t_phase, ref, st)
        self._log(t, ref, st, out, dt_sim)
        self._publish(out.motor_speeds)

def main():
    rclpy.init()
    node = HarnessFlight()
    code = 0 # safety abort code is 2
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit): 
        # _finish() explicitly raises SystemExit
        code = 2 if node.abort else 0
    node.destroy_node()
    rclpy.try_shutdown()
    sys.exit(code)

if __name__ == '__main__':
    main()