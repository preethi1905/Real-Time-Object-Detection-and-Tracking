#!/usr/bin/env python3
"""
detector_node.py
ROS 2 node that subscribes to /camera/image_raw, performs object detection
using YOLOv8 (PyTorch or ONNX), and publishes vision_msgs/Detection2DArray on /detections.
"""

import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge

# KITTI 3-Class Mapping
KITTI_CLASSES = {0: "Car", 1: "Pedestrian", 2: "Cyclist"}
# COCO to KITTI fallback mapping when using standard pretrained YOLOv8n
COCO_TO_KITTI = {
    2: 0,   # car -> Car
    5: 0,   # bus -> Car
    7: 0,   # truck -> Car
    0: 1,   # person -> Pedestrian
    1: 2,   # bicycle -> Cyclist
    3: 2,   # motorcycle -> Cyclist
}


class DetectorNode(Node):
    def __init__(self):
        super().__init__('detector_node')

        # Parameters
        self.declare_parameter('model_path', 'runs/detect/train/weights/best.pt')
        self.declare_parameter('fallback_model', 'yolov8n.pt')
        self.declare_parameter('conf_threshold', 0.25)
        self.declare_parameter('iou_threshold', 0.45)
        self.declare_parameter('device', 'cpu')  # 'cpu' or '0' for CUDA
        self.declare_parameter('imgsz', 640)

        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.fallback_model = self.get_parameter('fallback_model').get_parameter_value().string_value
        self.conf_thr = self.get_parameter('conf_threshold').get_parameter_value().double_value
        self.iou_thr = self.get_parameter('iou_threshold').get_parameter_value().double_value
        self.device = self.get_parameter('device').get_parameter_value().string_value
        self.imgsz = self.get_parameter('imgsz').get_parameter_value().integer_value

        self.bridge = CvBridge()
        self.model = None
        self.is_custom_kitti = False

        self._init_detector()

        # Subscribers & Publishers
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        self.det_publisher = self.create_publisher(Detection2DArray, '/detections', 10)

        self.get_logger().info("DetectorNode successfully initialized.")

    def _init_detector(self):
        from ultralytics import YOLO

        target_model = self.model_path
        if not os.path.exists(target_model):
            self.get_logger().warn(
                f"Model file '{target_model}' not found. Falling back to '{self.fallback_model}'..."
            )
            target_model = self.fallback_model
            self.is_custom_kitti = False
        else:
            self.get_logger().info(f"Loading custom fine-tuned KITTI model: {target_model}")
            self.is_custom_kitti = True

        try:
            self.model = YOLO(target_model)
            self.get_logger().info(f"YOLO model loaded successfully on device='{self.device}'.")
        except Exception as e:
            self.get_logger().error(f"Failed to load YOLO model: {e}")
            raise e

    def image_callback(self, msg: Image):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge conversion error: {e}")
            return

        # Perform YOLOv8 inference
        results = self.model.predict(
            source=cv_img,
            conf=self.conf_thr,
            iou=self.iou_thr,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False
        )

        # Build vision_msgs/Detection2DArray
        det_array_msg = Detection2DArray()
        det_array_msg.header = msg.header  # Preserve timestamp and frame_id for synchronization

        if results and len(results) > 0:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy().tolist()  # [x1, y1, x2, y2]
                conf = float(boxes.conf[i].cpu().numpy())
                cls_raw = int(boxes.cls[i].cpu().numpy())

                # Map class based on whether custom KITTI model or COCO pretrained model is used
                if self.is_custom_kitti:
                    if cls_raw not in KITTI_CLASSES:
                        continue
                    kitti_cls_id = cls_raw
                    class_name = KITTI_CLASSES[cls_raw]
                else:
                    if cls_raw not in COCO_TO_KITTI:
                        continue  # Skip classes that are not Car, Pedestrian, or Cyclist
                    kitti_cls_id = COCO_TO_KITTI[cls_raw]
                    class_name = KITTI_CLASSES[kitti_cls_id]

                x1, y1, x2, y2 = xyxy
                w = max(0.0, x2 - x1)
                h = max(0.0, y2 - y1)
                cx = x1 + w / 2.0
                cy = y1 + h / 2.0

                detection = Detection2D()
                detection.header = msg.header
                detection.id = f"{class_name}_{kitti_cls_id}"

                # Bounding Box 2D
                detection.bbox.center.position.x = float(cx)
                detection.bbox.center.position.y = float(cy)
                detection.bbox.center.theta = 0.0
                detection.bbox.size_x = float(w)
                detection.bbox.size_y = float(h)

                # Hypothesis
                hyp = ObjectHypothesisWithPose()
                # Support Humble vision_msgs structure
                if hasattr(hyp, 'hypothesis'):
                    hyp.hypothesis.class_id = str(kitti_cls_id)
                    hyp.hypothesis.score = float(conf)
                else:
                    # In earlier vision_msgs releases
                    setattr(hyp, 'id', str(kitti_cls_id))
                    setattr(hyp, 'score', float(conf))

                detection.results.append(hyp)
                det_array_msg.detections.append(detection)

        self.det_publisher.publish(det_array_msg)


def main(args=None):
    rclpy.init(args=args)
    node = DetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
