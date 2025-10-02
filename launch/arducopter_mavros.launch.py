from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # MAVROS - UAV 1
        Node(
            package='mavros',
            executable='mavros_node',
            namespace='uav1',
            arguments=[
                '--ros-args',
                '-p', 'fcu_url:=udp://:14550@0.0.0.0:14550',
                '-p', 'system_id:=255',
                '-p', 'target_system_id:=1',
                '-p', 'target_component_id:=1'
            ]
        ),

        # MAVROS - UAV 2
        Node(
            package='mavros',
            executable='mavros_node',
            namespace='uav2',
            arguments=[
                '--ros-args',
                '-p', 'fcu_url:=udp://:14551@0.0.0.0:14551',
                '-p', 'system_id:=255',
                '-p', 'target_system_id:=2',
                '-p', 'target_component_id:=1'
            ]
        ),

        # Visualizador - UAV 1
        Node(
            package='dk450',
            executable='MavrosVisualizer',
            namespace='uav1',
            parameters=[{'drone_ns': 'uav1'}]
        ),

        # Visualizador - UAV 2
        Node(
            package='dk450',
            executable='MavrosVisualizer',
            namespace='uav2',
            parameters=[{'drone_ns': 'uav2'}]
        ),

        # Agrega RViz
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen'
        ),
    ])

