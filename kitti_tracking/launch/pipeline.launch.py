import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Declare launch arguments
    sequence_path_arg = DeclareLaunchArgument(
        'sequence_path',
        default_value='data/kitti/tracking/training/image_02/0000',
        description='Path to the KITTI tracking image sequence directory'
    )

    publish_rate_arg = DeclareLaunchArgument(
        'publish_rate',
        default_value='10.0',
        description='Rate in Hz to publish camera frames'
    )

    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='runs/detect/train/weights/best.pt',
        description='Path to trained YOLOv8 model weights (.pt or .onnx)'
    )

    fallback_model_arg = DeclareLaunchArgument(
        'fallback_model',
        default_value='yolov8n.pt',
        description='Fallback pretrained YOLOv8 model weights'
    )

    conf_threshold_arg = DeclareLaunchArgument(
        'conf_threshold',
        default_value='0.25',
        description='Detection confidence threshold'
    )

    device_arg = DeclareLaunchArgument(
        'device',
        default_value='cpu',
        description='Inference compute device (cpu or 0 for cuda)'
    )

    max_age_arg = DeclareLaunchArgument(
        'max_age',
        default_value='5',
        description='Maximum frames to preserve a track without detections'
    )

    min_hits_arg = DeclareLaunchArgument(
        'min_hits',
        default_value='2',
        description='Minimum detections before track confirmation'
    )

    iou_threshold_arg = DeclareLaunchArgument(
        'iou_threshold',
        default_value='0.3',
        description='IoU threshold for Hungarian matching'
    )

    # Node 1: Image Publisher Node
    image_publisher_node = Node(
        package='kitti_tracking',
        executable='image_publisher_node',
        name='image_publisher_node',
        output='screen',
        parameters=[{
            'sequence_path': LaunchConfiguration('sequence_path'),
            'publish_rate': LaunchConfiguration('publish_rate'),
            'loop': True
        }]
    )

    # Node 2: Detector Node
    detector_node = Node(
        package='kitti_tracking',
        executable='detector_node',
        name='detector_node',
        output='screen',
        parameters=[{
            'model_path': LaunchConfiguration('model_path'),
            'fallback_model': LaunchConfiguration('fallback_model'),
            'conf_threshold': LaunchConfiguration('conf_threshold'),
            'device': LaunchConfiguration('device'),
            'imgsz': 640
        }]
    )

    # Node 3: Tracker Node
    tracker_node = Node(
        package='kitti_tracking',
        executable='tracker_node',
        name='tracker_node',
        output='screen',
        parameters=[{
            'max_age': LaunchConfiguration('max_age'),
            'min_hits': LaunchConfiguration('min_hits'),
            'iou_threshold': LaunchConfiguration('iou_threshold'),
            'draw_trajectories': True,
            'draw_velocities': True
        }]
    )

    return LaunchDescription([
        sequence_path_arg,
        publish_rate_arg,
        model_path_arg,
        fallback_model_arg,
        conf_threshold_arg,
        device_arg,
        max_age_arg,
        min_hits_arg,
        iou_threshold_arg,
        image_publisher_node,
        detector_node,
        tracker_node
    ])
