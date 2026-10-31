import os
from glob import glob
from setuptools import setup, find_packages

package_name = 'kitti_tracking'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Preethi Prabhakaran',
    maintainer_email='preethi@example.com',
    description='Real-Time Object Detection and Multi-Object Tracking Pipeline with ROS 2 and KITTI',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'image_publisher_node = kitti_tracking.image_publisher_node:main',
            'detector_node = kitti_tracking.detector_node:main',
            'tracker_node = kitti_tracking.tracker_node:main',
        ],
    },
)
