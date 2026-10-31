#!/usr/bin/env python3
"""
image_publisher_node.py
ROS 2 node that reads KITTI tracking image sequences in sequential order at 10 Hz
and publishes sensor_msgs/Image on /camera/image_raw using cv_bridge.
"""

import os
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class ImagePublisherNode(Node):
    def __init__(self):
        super().__init__('image_publisher_node')

        # Declare ROS 2 parameters
        self.declare_parameter('sequence_path', 'data/kitti/tracking/training/image_02/0000')
        self.declare_parameter('publish_rate', 10.0)  # KITTI default is ~10 Hz
        self.declare_parameter('loop', True)
        self.declare_parameter('frame_id', 'camera_optical_frame')

        self.sequence_path = self.get_parameter('sequence_path').get_parameter_value().string_value
        self.publish_rate = self.get_parameter('publish_rate').get_parameter_value().double_value
        self.loop = self.get_parameter('loop').get_parameter_value().bool_value
        self.frame_id = self.get_parameter('frame_id').get_parameter_value().string_value

        self.bridge = CvBridge()
        self.image_files = []
        self.current_idx = 0

        # Scan and sort sequence images
        self._load_sequence()

        # Publisher
        self.publisher_ = self.create_publisher(Image, '/camera/image_raw', 10)

        # Timer
        timer_period = 1.0 / max(self.publish_rate, 0.1)
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.get_logger().info(
            f"ImagePublisherNode initialized: Publishing {len(self.image_files)} frames from "
            f"'{self.sequence_path}' at {self.publish_rate} Hz (loop={self.loop})"
        )

    def _load_sequence(self):
        if not os.path.exists(self.sequence_path):
            self.get_logger().warn(
                f"Sequence path '{self.sequence_path}' does not exist! "
                f"Please ensure KITTI sequence images are in place."
            )
            return

        valid_exts = ('.png', '.jpg', '.jpeg')
        files = [
            f for f in os.listdir(self.sequence_path)
            if f.lower().endswith(valid_exts)
        ]
        # Sort numerically: 000000.png, 000001.png, etc.
        files.sort(key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else x)
        self.image_files = [os.path.join(self.sequence_path, f) for f in files]

        if not self.image_files:
            self.get_logger().warn(f"No valid image files found in {self.sequence_path}")

    def timer_callback(self):
        if not self.image_files:
            return

        if self.current_idx >= len(self.image_files):
            if self.loop:
                self.current_idx = 0
                self.get_logger().info("Sequence completed. Looping back to start.")
            else:
                self.get_logger().info("Sequence published completely. Halting publication.")
                self.timer.cancel()
                return

        img_path = self.image_files[self.current_idx]
        cv_img = cv2.imread(img_path)

        if cv_img is None:
            self.get_logger().error(f"Failed to read image at {img_path}")
            self.current_idx += 1
            return

        # Convert OpenCV image to ROS 2 Image message
        msg = self.bridge.cv2_to_imgmsg(cv_img, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id

        self.publisher_.publish(msg)

        if self.current_idx % 50 == 0:
            self.get_logger().info(
                f"Publishing frame {self.current_idx}/{len(self.image_files)}: {os.path.basename(img_path)}"
            )

        self.current_idx += 1


def main(args=None):
    rclpy.init(args=args)
    node = ImagePublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
