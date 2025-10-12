#NOTE: This file is directly copied from: https://github.com/kobuki-base/kobuki_ros/blob/devel/kobuki_node/launch/kobuki_node-launch.py
# and the remappings were added
import os

import ament_index_python
import launch
import launch_ros.actions
import launch_ros.substitutions

import yaml

# TODO: This file should be removed when the parameter file loading is fixed.

def generate_launch_description():
    share_dir = ament_index_python.packages.get_package_share_directory('tb2_system')
    # There are two different ways to pass parameters to a non-composed node;
    # either by specifying the path to the file containing the parameters, or by
    # passing a dictionary containing the key -> value pairs of the parameters.
    # When starting a *composed* node on the other hand, only the dictionary
    # style is supported.  To keep the code between the non-composed and
    # composed launch file similar, we use that style here as well.
    params_file = os.path.join(share_dir, 'config', 'optitrack', 'kobuki_params.yaml')
    with open(params_file, 'r') as f:
        params = yaml.safe_load(f)['kobuki_ros_node']['ros__parameters']

    remappings = [
        ('commands/velocity', 'cmd_vel'),
    ]
    # Add log level to params
    params['log_level'] = 'debug'
    kobuki_ros_node = launch_ros.actions.Node(package='kobuki_node',
                                              executable='kobuki_ros_node',
                                              output='both',
                                              parameters=[params],
                                              remappings=remappings)

    return launch.LaunchDescription([kobuki_ros_node])