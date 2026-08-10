from setuptools import setup

package_name = 'cf_tracking'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name, package_name + '.controllers'],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config',
            ['config/vehicle.yaml', 'config/gz_bridge_tracking.yaml']),
        ('share/' + package_name + '/launch',
            ['launch/track_reference.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    
    maintainer='Rakshith Ajetaravi',
    maintainer_email='rakshithajetaravi@gmail.com',
    description='Crazyflie traj tracking sim with different controllers',
    license='MIT',
    
    entry_points={
        'console_scripts': [
            'harness_flight = cf_tracking.harness_flight:main',
        ],
    },
)
