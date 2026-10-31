#!/usr/bin/env python3
"""
tracker_node.py
ROS 2 node that subscribes to /detections and /camera/image_raw, runs the
custom Kalman-Hungarian Tracker, publishes tracked objects on /tracks,
and publishes visually annotated tracking frames on /image_tracked.
"""

from collections import deque
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge

from .tracker import Tracker

KITTI_CLASSES = {0: "Car", 1: "Pedestrian", 2: "Cyclist"}

# Distinct color palette for consistent track ID visualization
COLOR_PALETTE = [
    (255, 56, 56),   (255, 157, 151), (255, 112, 31),  (255, 178, 29),
    (207, 210, 49),  (72, 249, 10),   (146, 204, 23),  (61, 219, 134),
    (26, 147, 65),   (35, 181, 211),  (18, 123, 243),  (19, 71, 237),
    (67, 68, 245),   (133, 47, 239),  (203, 56, 241),  (245, 66, 179)
]


def get_color(track_id):
    return COLOR_PALETTE[track_id % len(COLOR_PALETTE)]


class TrackerNode(Node):
    def __init__(self):
        super().__init__('tracker_node')

        # Parameters
        self.declare_parameter('max_age', 5)
        self.declare_parameter('min_hits', 2)
        self.declare_parameter('iou_threshold', 0.3)
        self.declare_parameter('class_aware', False)
        self.declare_parameter('draw_trajectories', True)
        self.declare_parameter('draw_velocities', True)

        max_age = self.get_parameter('max_age').get_parameter_value().integer_value
        min_hits = self.get_parameter('min_hits').get_parameter_value().integer_value
        iou_thr = self.get_parameter('iou_threshold').get_parameter_value().double_value
        class_aware = self.get_parameter('class_aware').get_parameter_value().bool_value
        self.draw_trajectories = self.get_parameter('draw_trajectories').get_parameter_value().bool_value
        self.draw_velocities = self.get_parameter('draw_velocities').get_parameter_value().bool_value

        self.bridge = CvBridge()
        self.tracker = Tracker(
            max_age=max_age,
            min_hits=min_hits,
            iou_thr=iou_thr,
            class_aware=class_aware
        )

        # Buffer for images (timestamp -> cv_image)
        self.image_buffer = {}
        self.latest_cv_image = None
        self.track_history = {}  # track_id -> deque of (cx, cy)
        self.frame_count = 0

        # Subscribers
        self.sub_image = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        self.sub_dets = self.create_subscription(
            Detection2DArray,
            '/detections',
            self.detections_callback,
            10
        )

        # Publishers
        self.pub_tracks = self.create_publisher(Detection2DArray, '/tracks', 10)
        self.pub_tracked_image = self.create_publisher(Image, '/image_tracked', 10)

        self.get_logger().info(
            f"TrackerNode initialized: max_age={max_age}, min_hits={min_hits}, "
            f"iou_thr={iou_thr}, class_aware={class_aware}"
        )

    def image_callback(self, msg: Image):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            t_key = (msg.header.stamp.sec, msg.header.stamp.nanosec)
            self.image_buffer[t_key] = cv_img
            self.latest_cv_image = cv_img

            # Maintain buffer size <= 30
            if len(self.image_buffer) > 30:
                oldest_key = min(self.image_buffer.keys())
                del self.image_buffer[oldest_key]
        except Exception as e:
            self.get_logger().error(f"Image callback cv_bridge error: {e}")

    def detections_callback(self, msg: Detection2DArray):
        self.frame_count += 1
        t_key = (msg.header.stamp.sec, msg.header.stamp.nanosec)

        # 1. Parse incoming detections
        detections = []
        for det in msg.detections:
            cx = det.bbox.center.position.x
            cy = det.bbox.center.position.y
            w = det.bbox.size_x
            h = det.bbox.size_y
            x1 = cx - w / 2.0
            y1 = cy - h / 2.0
            x2 = cx + w / 2.0
            y2 = cy + h / 2.0

            score = 1.0
            class_id = 0
            if det.results:
                hyp = det.results[0]
                if hasattr(hyp, 'hypothesis'):
                    class_id = int(hyp.hypothesis.class_id)
                    score = float(hyp.hypothesis.score)
                elif hasattr(hyp, 'id'):
                    class_id = int(hyp.id)
                    score = float(hyp.score)

            detections.append([x1, y1, x2, y2, score, class_id])

        # 2. Step the Kalman + Hungarian Tracker
        active_tracks = self.tracker.step(detections)

        # 3. Build & Publish /tracks (vision_msgs/Detection2DArray)
        tracks_msg = Detection2DArray()
        tracks_msg.header = msg.header

        for track in active_tracks:
            tid, bbox, cid, score, vel = track
            x1, y1, x2, y2 = bbox
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            cx = x1 + w / 2.0
            cy = y1 + h / 2.0

            det2d = Detection2D()
            det2d.header = msg.header
            det2d.id = str(tid)

            det2d.bbox.center.position.x = float(cx)
            det2d.bbox.center.position.y = float(cy)
            det2d.bbox.center.theta = 0.0
            det2d.bbox.size_x = float(w)
            det2d.bbox.size_y = float(h)

            hyp = ObjectHypothesisWithPose()
            if hasattr(hyp, 'hypothesis'):
                hyp.hypothesis.class_id = str(cid)
                hyp.hypothesis.score = float(score)
            else:
                setattr(hyp, 'id', str(cid))
                setattr(hyp, 'score', float(score))

            det2d.results.append(hyp)
            tracks_msg.detections.append(det2d)

        self.pub_tracks.publish(tracks_msg)

        # 4. Render & Publish /image_tracked
        frame = self.image_buffer.pop(t_key, self.latest_cv_image)
        if frame is not None:
            vis_img = frame.copy()
            self._render_tracks(vis_img, active_tracks)

            out_msg = self.bridge.cv2_to_imgmsg(vis_img, encoding="bgr8")
            out_msg.header = msg.header
            self.pub_tracked_image.publish(out_msg)

    def _render_tracks(self, img, active_tracks):
        # Update history and draw trajectories
        active_ids = set()
        for track in active_tracks:
            tid, bbox, cid, score, vel = track
            active_ids.add(tid)
            x1, y1, x2, y2 = map(int, bbox)
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            color = get_color(tid)

            if tid not in self.track_history:
                self.track_history[tid] = deque(maxlen=30)
            self.track_history[tid].append((cx, cy))

            # Draw trajectory trail
            if self.draw_trajectories and len(self.track_history[tid]) > 1:
                pts = list(self.track_history[tid])
                for j in range(1, len(pts)):
                    cv2.line(img, pts[j - 1], pts[j], color, 2)

            # Draw Bounding Box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # Draw Velocity vector if available
            if self.draw_velocities and vel:
                vx, vy = vel
                end_x = int(cx + vx * 2.0)
                end_y = int(cy + vy * 2.0)
                cv2.arrowedLine(img, (cx, cy), (end_x, end_y), (0, 255, 255), 2, tipLength=0.3)

            # Draw Label Tag
            cls_name = KITTI_CLASSES.get(cid, f"Class {cid}")
            label = f"ID:{tid} {cls_name} {score:.2f}"
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(img, (x1, y1 - 20), (x1 + t_size[0] + 6, y1), color, -1)
            cv2.putText(
                img, label, (x1 + 3, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
            )

        # Clean up stale track histories
        stale_ids = [tid for tid in self.track_history if tid not in active_ids]
        for tid in stale_ids:
            if len(self.track_history[tid]) > 0:
                self.track_history[tid].popleft()
            if len(self.track_history[tid]) == 0:
                del self.track_history[tid]

        # Draw HUD overlay
        hud_text = f"Frame: {self.frame_count} | Active Tracks: {len(active_tracks)}"
        cv2.putText(
            img, hud_text, (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA
        )


def main(args=None):
    rclpy.init(args=args)
    node = TrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
