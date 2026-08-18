#!/usr/bin/env python3
"""Launch the onboard person servo node and its deadman watchdog.

Mirrors gremsy_ros2/launch/spirit_triage.launch.py: the drone is chosen from
$ROBOT_NAME (or the `drone` arg), which selects config/<drone>.yaml. Nothing is
hardcoded in the build.

Everything here runs inside the drone's own ROS_DOMAIN_ID -- there is no
basestation involvement and no domain bridge entry to add.

Args:
  drone          Robot name; selects config/<drone>.yaml (default $ROBOT_NAME|spiritnx3).
  namespace      Node namespace (default /<drone>).
  config_file    Override the per-drone config file path.
  servo_backend  Override the backend (dry_run|track_touch|angle|rate|single_axis_rate).
  deadman        Whether to start the watchdog (default true).
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _launch_setup(context, *args, **kwargs):
    drone = LaunchConfiguration('drone').perform(context)
    namespace = LaunchConfiguration('namespace').perform(context) or f'/{drone}'
    servo_backend = LaunchConfiguration('servo_backend').perform(context)

    config_file = LaunchConfiguration('config_file').perform(context)
    if not config_file:
        share = get_package_share_directory('spirit_person_servo')
        config_file = os.path.join(share, 'config', f'{drone}.yaml')

    parameters = []
    if os.path.isfile(config_file):
        parameters.append(config_file)
    else:
        print(f'[person_servo] WARNING: config file not found: {config_file} '
              f'(using node defaults).')

    overrides = {'robot_name': drone}
    if servo_backend:
        overrides['servo_backend'] = servo_backend
    parameters.append(overrides)

    servo = Node(
        package='spirit_person_servo',
        executable='person_servo_node',
        name='person_servo_node',
        namespace=namespace,
        output='screen',
        parameters=parameters,
    )
    deadman = Node(
        package='spirit_person_servo',
        executable='gimbal_deadman_node',
        name='gimbal_deadman_node',
        namespace=namespace,
        output='screen',
        parameters=[{
            'gimbal_namespace': f'/{drone}/gremsy',
            'state_topic': f'{namespace}/person_servo_node/state',
        }],
        condition=IfCondition(LaunchConfiguration('deadman')),
    )
    return [servo, deadman]


def generate_launch_description():
    default_drone = os.environ.get('ROBOT_NAME', 'spiritnx3')
    return LaunchDescription([
        DeclareLaunchArgument('drone', default_value=default_drone),
        DeclareLaunchArgument('namespace', default_value=''),
        DeclareLaunchArgument('config_file', default_value=''),
        DeclareLaunchArgument('servo_backend', default_value=''),
        DeclareLaunchArgument('deadman', default_value='true'),
        OpaqueFunction(function=_launch_setup),
    ])
