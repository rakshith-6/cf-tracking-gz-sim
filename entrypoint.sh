#!/bin/bash

# Exit immediately if the exit code is non-zero
set -e

# Source ROS2 installation
source /opt/ros/jazzy/setup.bash

# Overlay workspace if it has been built
if [ -f /ws/install/setup.bash ]; then
    source /ws/install/setup.bash
fi
exec "$@"
