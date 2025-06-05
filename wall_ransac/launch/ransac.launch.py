#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    # Launch引数の宣言
    min_distance_arg = DeclareLaunchArgument(
        'min_distance_threshold',
        default_value='0.1',
        description='最小距離閾値 (m)'
    )
    
    max_distance_arg = DeclareLaunchArgument(
        'max_distance_threshold', 
        default_value='5.0',
        description='最大距離閾値 (m)'
    )
    
    ransac_iterations_arg = DeclareLaunchArgument(
        'ransac_iterations',
        default_value='1000',
        description='RANSAC反復回数'
    )
    
    ransac_threshold_arg = DeclareLaunchArgument(
        'ransac_threshold',
        default_value='0.05',
        description='RANSAC距離閾値 (m)'
    )
    
    min_points_arg = DeclareLaunchArgument(
        'min_points_for_wall',
        default_value='10', 
        description='壁として認識する最小ポイント数'
    )
    
    wall_distance_threshold_arg = DeclareLaunchArgument(
        'wall_distance_threshold',
        default_value='1.0',
        description='壁との距離警告閾値 (m)'
    )
    
    # 壁検知ノード
    wall_detection_node = Node(
        package='wall_ransac',  # パッケージ名に置き換えてください
        executable='ransac_node',
        name='ransac_node',
        output='screen',
        parameters=[{
            'min_distance_threshold': LaunchConfiguration('min_distance_threshold'),
            'max_distance_threshold': LaunchConfiguration('max_distance_threshold'),
            'ransac_iterations': LaunchConfiguration('ransac_iterations'),
            'ransac_threshold': LaunchConfiguration('ransac_threshold'),
            'min_points_for_wall': LaunchConfiguration('min_points_for_wall'),
            'wall_distance_threshold': LaunchConfiguration('wall_distance_threshold'),
        }]
    )
    
    return LaunchDescription([
        min_distance_arg,
        max_distance_arg,
        ransac_iterations_arg,
        ransac_threshold_arg,
        min_points_arg,
        wall_distance_threshold_arg,
        wall_detection_node,
    ])
