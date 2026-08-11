"""
Launch file for orchestration of the flight experiment: 

Examples usage(inside the container):
  ros2 launch cf_tracking track_reference.launch.py mode:=open_loop duration:=3.0
  ros2 launch cf_tracking track_reference.launch.py mode:=hover duration:=20.0
  ros2 launch cf_tracking track_reference.launch.py controller:=lee \
      primitive:=circle max_speed:=0.5 duration:=30.0 gui:=true
"""
import os

'''
- LaunchDescription: list of instructions to excecute
- Launch actions:
    - DeclareLaunchArguments: define command line arguments
    - EmitEvent: manually send/trigger an event
    - ExecuteProcess: lets launch file run an external command/program
    - OpaqueFunction: to run your own python fun/code during the launch process
    - RegisterEventHandler: watch for an event and perform some action
    - SetEnvironmentVariable: used to set environment settings for the launched 
                              process
- Launch event handler: we are using OnProcessExit event handler here to perform 
                        some action on process exit
- Launch events: we are using Shutdown event to stop lauch system
- Launch substitutes: using LaunchConfiguration(arg_name) we can get the launch 
                      argument value
- Launch ROS modules: 
    - actions: Node is used to start a ROS 2 node
    - parameter descriptions: ParameterValue is used to define parameter type
                              for ROS 2
- get_package_share_directory: Helps in finding files inside a ROS 2 package
(once we find the shared directory path using package name, we can find any 
other files easily using this path)
'''
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, EmitEvent, ExecuteProcess,
                            OpaqueFunction, RegisterEventHandler,
                            SetEnvironmentVariable)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory

'''
WindEffects SDF Block: adds wind effect plugin/configuration to the world, so that 
we can use the modified world to start the gazebo sim. {wx}, {wy}, and {wz} are 
placeholders, later we can do _WIND_SDF.format(...) to assign values. 
- applied wind force = FASF(=0.2)*calculated wind force(we can tune FASF)
- For both wind magnitude and direction(horizontal and vertical)base wind * +/- 10% 
  will be amplitude of the sine wave with some period + we add gaussian noise
'''
_WIND_SDF = """
    <wind>
      <linear_velocity>{wx} {wy} {wz}</linear_velocity>
    </wind>
    <plugin
      filename="gz-sim-wind-effects-system"
      name="gz::sim::systems::WindEffects">
      <force_approximation_scaling_factor>0.2</force_approximation_scaling_factor>
      <horizontal>
        <magnitude>
          <time_for_rise>2</time_for_rise>
          <sin>
            <amplitude_percent>0.10</amplitude_percent>
            <period>10</period>
          </sin>
          <noise type="gaussian">
            <mean>0</mean>
            <stddev>0.02</stddev>
          </noise>
        </magnitude>
        <direction>
          <time_for_rise>5</time_for_rise>
          <sin>
            <amplitude>5</amplitude>
            <period>15</period>
          </sin>
          <noise type="gaussian">
            <mean>0</mean>
            <stddev>0.02</stddev>
          </noise>
        </direction>
      </horizontal>
      <vertical>
        <noise type="gaussian">
          <mean>0</mean>
          <stddev>0.01</stddev>
        </noise>
      </vertical>
    </plugin>
"""

'''
- _start_sim() is called by OpaqueFunction during launch excecution
- In a ROS 2 launch file, context contains information about the 
  current launch configuration/arguments supplied by the user
- re/regular expression is used to find and replace things inside
  the SDF files
- shutil provides functions for working with files and directories
- If the user requests any domain randomization then, temp model 
  directory is created, copy the original models there, writes a
  modified sdf(using re.sub()) as required and use it in the sim 
- If RTF or wx, wy or wz is requested by the user, we modify the
  the world sdf file. Sim starts with zero wind if wind_onset_delay 
  is requested, then later harness switches the commanded vector on 
  via the runtime /world/cf_tracking/wind topic
- This function starts Gazebo. '-s' tag in gz sim means headless or
  without GUI run
'''
def _start_sim(context):
    import re
    import shutil

    gui = context.launch_configurations.get('gui', 'false').lower() == 'true'
    world = '/sim/worlds/cf_tracking.sdf'
    
    wx = float(context.launch_configurations.get('wind_x', '0.0'))
    wy = float(context.launch_configurations.get('wind_y', '0.0'))
    wz = float(context.launch_configurations.get('wind_z', '0.0'))
    mass_scale = float(context.launch_configurations.get('mass_scale', '1.0'))
    inertia_scale = float(
        context.launch_configurations.get('inertia_scale', '1.0'))
    
    resource_path = '/sim/models'
    if mass_scale != 1.0 or inertia_scale != 1.0:
        gen_models = '/tmp/models_gen_tracking'
        shutil.rmtree(gen_models, ignore_errors=True)
        shutil.copytree('/sim/models', gen_models)
        model_path = os.path.join(gen_models, 'crazyflie_tracking',
                                  'model.sdf')
        with open(model_path) as f:
            model = f.read()
        model = re.sub(r'<mass>0.0283</mass>',
                       f'<mass>{0.0283 * mass_scale:.6f}</mass>', model)
        for tag in ('ixx', 'iyy', 'izz', 'ixy', 'ixz', 'iyz'):
            model = re.sub(
                rf'<{tag}>([-0-9.eE]+)</{tag}>',
                lambda g: f'<{tag}>{float(g.group(1)) * inertia_scale:.9e}'
                          f'</{tag}>', model)
        with open(model_path, 'w') as f:
            f.write(model)
        resource_path = f'{gen_models}:/sim/models'
    
    # We read rtf and wind onset delay from command line arguments
    rtf = float(context.launch_configurations.get('rtf', '1.0'))
    onset = float(context.launch_configurations.get('wind_onset_delay',
                                                    '-1.0'))
    if rtf != 1.0 or wx or wy or wz:
        with open(world) as f:
            sdf = f.read()
        if wx or wy or wz:
            w0 = (0.0, 0.0, 0.0) if onset >= 0.0 else (wx, wy, wz)
            sdf = sdf.replace('<model name="ground_plane">',
                              _WIND_SDF.format(wx=w0[0], wy=w0[1],
                                               wz=w0[2])
                              + '    <model name="ground_plane">')
        if rtf != 1.0:
            sdf = re.sub(r'<real_time_factor>[^<]*</real_time_factor>',
                         f'<real_time_factor>{rtf}</real_time_factor>', sdf)
        world = '/tmp/cf_tracking_gen.sdf'
        with open(world, 'w') as f:
            f.write(sdf)
    cmd = ['gz', 'sim', '-r', '-v', '1', world]
    if not gui:
        cmd.insert(2, '-s')
    return [ExecuteProcess(
        cmd=cmd, additional_env={'GZ_SIM_RESOURCE_PATH': resource_path},
        output='screen')]

'''
Launch description function:

- Gets the package shared directory and then .yaml file
- Starts the harness ROS 2 Node
- Returns the LaunchDescription(calling _start_sim will and return, starting
  gz-ros-bridge Node, starting flight/harness Node and shut everything down)
'''
def generate_launch_description():
    pkg_share = get_package_share_directory('cf_tracking')
    bridge_cfg = os.path.join(pkg_share, 'config', 'gz_bridge_tracking.yaml')

    flight = Node(
        package='cf_tracking',
        executable='harness_flight',
        parameters=[{
            'use_sim_time': True,
            'controller': LaunchConfiguration('controller'),
            'mode': LaunchConfiguration('mode'),
            'primitive': LaunchConfiguration('primitive'),
            'yaw_mode': LaunchConfiguration('yaw_mode'),
            'seed': ParameterValue(LaunchConfiguration('seed'),
                                   value_type=int),
            'max_speed': ParameterValue(LaunchConfiguration('max_speed'),
                                        value_type=float),
            'duration': ParameterValue(LaunchConfiguration('duration'),
                                       value_type=float),
            'altitude': ParameterValue(LaunchConfiguration('altitude'),
                                       value_type=float),
            'radius': ParameterValue(LaunchConfiguration('radius'),
                                     value_type=float),
            'box_xy': ParameterValue(LaunchConfiguration('box_xy'),
                                     value_type=float),
            'cmd_rate': ParameterValue(LaunchConfiguration('cmd_rate'),
                                       value_type=float),
            'out_dir': LaunchConfiguration('out_dir'),
            'run_name': LaunchConfiguration('run_name'),
            'wind_x': ParameterValue(LaunchConfiguration('wind_x'),
                                     value_type=float),
            'wind_y': ParameterValue(LaunchConfiguration('wind_y'),
                                     value_type=float),
            'wind_z': ParameterValue(LaunchConfiguration('wind_z'),
                                     value_type=float),
            'wind_onset_delay': ParameterValue(
                LaunchConfiguration('wind_onset_delay'), value_type=float),
            'mass_scale': ParameterValue(LaunchConfiguration('mass_scale'),
                                         value_type=float),
            'inertia_scale': ParameterValue(
                LaunchConfiguration('inertia_scale'), value_type=float)}],
        output='screen')

    return LaunchDescription([
        DeclareLaunchArgument('controller', default_value='lee'),
        DeclareLaunchArgument('mode', default_value='track'),
        DeclareLaunchArgument('primitive', default_value='circle'),
        DeclareLaunchArgument('yaw_mode', default_value='fixed'),
        DeclareLaunchArgument('seed', default_value='0'),
        DeclareLaunchArgument('max_speed', default_value='0.5'),
        DeclareLaunchArgument('duration', default_value='30.0'),
        DeclareLaunchArgument('altitude', default_value='1.0'),
        DeclareLaunchArgument('radius', default_value='0.6'),
        DeclareLaunchArgument('box_xy', default_value='3.0'),
        DeclareLaunchArgument('rtf', default_value='1.0'),
        DeclareLaunchArgument('wind_onset_delay', default_value='-1.0'),
        DeclareLaunchArgument('cmd_rate', default_value='250.0'),
        DeclareLaunchArgument('out_dir', default_value='/data/tracking_runs'),
        DeclareLaunchArgument('run_name', default_value=''),
        DeclareLaunchArgument('gui', default_value='false'),
        DeclareLaunchArgument('wind_x', default_value='0.0'),
        DeclareLaunchArgument('wind_y', default_value='0.0'),
        DeclareLaunchArgument('wind_z', default_value='0.0'),
        DeclareLaunchArgument('mass_scale', default_value='1.0'),
        DeclareLaunchArgument('inertia_scale', default_value='1.0'),
        SetEnvironmentVariable('GZ_SIM_RESOURCE_PATH', '/sim/models'),

        OpaqueFunction(function=_start_sim),

        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            parameters=[{'config_file': bridge_cfg, 'use_sim_time': True}],
            output='screen'),

        flight,
        RegisterEventHandler(OnProcessExit(
            target_action=flight,
            on_exit=[EmitEvent(event=Shutdown(reason='flight complete'))])),
    ])