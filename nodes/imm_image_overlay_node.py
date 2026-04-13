#!/usr/bin/env python3
"""
IMM Prediction Overlay Node
============================
Subscribes to an observer drone's camera image, the predicted IMM trajectory,
and the current filtered position estimate, then reprojects the 3D trajectory
onto the 2D image plane and publishes an annotated image.

Usage (standalone):
    python3 imm_image_overlay_node.py --ros-args \
        -p uav_name:=uav1 \
        -p target_name:=uav3

Or import the node class and spin it from your own launcher.

Published topics:
    /<uav_name>/imm_overlay/compressed   (sensor_msgs/CompressedImage)

Subscribed topics:
    /<uav_name>/rgb/image_raw            (sensor_msgs/Image or CompressedImage)
    /<uav_name>/rgb/camera_info          (sensor_msgs/CameraInfo)
    /target/predicted_trajectory         (nav_msgs/Path)
    /target/odometry_estimate            (nav_msgs/Odometry)
    /target/triangulation_raw            (geometry_msgs/PointStamped)

TF required:
    <uav_name>/<world_frame_suffix> → <camera_frame>
"""

import os
import time
from pathlib import Path as FsPath
from typing import List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry, Path
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from tf2_ros import Buffer, TransformListener


def quat_to_rotmat(x: float, y: float, z: float, w: float) -> np.ndarray:
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array([
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
    ], dtype=np.float64)


class IMMImageOverlayNode(Node):
    def __init__(self) -> None:
        super().__init__('imm_image_overlay_node')

        # ── Load config ──────────────────────────────────────────────────
        _config_path = FsPath(__file__).resolve().parent.parent / 'config' / 'runtime_stack.yaml'
        _override_image_topic = ''
        if _config_path.exists():
            import yaml
            try:
                with open(_config_path, 'r', encoding='utf-8') as _f:
                    _cfg = yaml.safe_load(_f)
                    if _cfg and 'vision' in _cfg:
                        _override_image_topic = _cfg['vision'].get('observer_image_topic', '') or ''
            except Exception:
                pass

        # ── Parameters ───────────────────────────────────────────────────
        self.declare_parameter('uav_name', 'uav1')
        self.declare_parameter('target_name', 'uav3')
        self.declare_parameter('world_frame_suffix', 'world_origin')
        self.declare_parameter('tf_timeout_sec', 0.1)
        self.declare_parameter('processing_rate_hz', 10.0)
        self.declare_parameter('trajectory_color_start', [0, 255, 255])   # cyan (BGR)
        self.declare_parameter('trajectory_color_end', [0, 0, 255])       # red (BGR)
        self.declare_parameter('estimate_color', [255, 165, 0])           # orange (BGR)
        self.declare_parameter('raw_triangulation_color', [0, 255, 0])    # green (BGR)
        self.declare_parameter('point_radius', 6)
        self.declare_parameter('line_thickness', 2)
        self.declare_parameter('jpeg_quality', 75)
        self.declare_parameter('draw_raw_triangulation', True)
        self.declare_parameter('draw_velocity_arrow', True)

        self.uav_name = str(self.get_parameter('uav_name').value)
        self.target_name = str(self.get_parameter('target_name').value)
        self.world_frame_suffix = str(self.get_parameter('world_frame_suffix').value)
        self.tf_timeout_sec = float(self.get_parameter('tf_timeout_sec').value)
        self.processing_rate_hz = max(1.0, float(self.get_parameter('processing_rate_hz').value))
        self.traj_color_start = tuple(self.get_parameter('trajectory_color_start').value)
        self.traj_color_end = tuple(self.get_parameter('trajectory_color_end').value)
        self.estimate_color = tuple(self.get_parameter('estimate_color').value)
        self.raw_tri_color = tuple(self.get_parameter('raw_triangulation_color').value)
        self.point_radius = max(2, int(self.get_parameter('point_radius').value))
        self.line_thickness = max(1, int(self.get_parameter('line_thickness').value))
        self.jpeg_quality = int(self.get_parameter('jpeg_quality').value)
        self.draw_raw_triangulation = bool(self.get_parameter('draw_raw_triangulation').value)
        self.draw_velocity_arrow = bool(self.get_parameter('draw_velocity_arrow').value)

        # ── Topic names ──────────────────────────────────────────────────
        self.image_topic = _override_image_topic or f'/{self.uav_name}/rgb/image_raw'
        self._use_compressed = self.image_topic.endswith('/compressed')
        _img_base = self.image_topic.replace('/compressed', '').rsplit('/image_raw', 1)[0]
        self.camera_info_topic = f'{_img_base}/camera_info'
        self.predicted_path_topic = '/target/predicted_trajectory'
        self.estimate_odom_topic = '/target/odometry_estimate'
        self.raw_tri_topic = '/target/triangulation_raw'
        self.output_topic = f'/{self.uav_name}/imm_overlay/compressed'

        self.world_frame_id = f'{self.uav_name}/{self.world_frame_suffix}'

        # ── State ────────────────────────────────────────────────────────
        self.bridge = CvBridge()
        self.camera_info: Optional[CameraInfo] = None
        self.latest_image = None
        self.latest_path: Optional[Path] = None
        self.latest_estimate: Optional[Odometry] = None
        self.latest_raw_tri: Optional[PointStamped] = None
        self._last_image_time = time.time()

        # ── TF ───────────────────────────────────────────────────────────
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ── Publishers ───────────────────────────────────────────────────
        self.overlay_pub = self.create_publisher(CompressedImage, self.output_topic, 10)

        # ── Subscribers ──────────────────────────────────────────────────
        _sensor_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
        )
        if self._use_compressed:
            self.create_subscription(CompressedImage, self.image_topic, self._image_cb, _sensor_qos)
        else:
            self.create_subscription(Image, self.image_topic, self._image_cb, _sensor_qos)
        self.create_subscription(CameraInfo, self.camera_info_topic, self._cam_info_cb, 10)
        self.create_subscription(Path, self.predicted_path_topic, self._path_cb, 10)
        self.create_subscription(Odometry, self.estimate_odom_topic, self._estimate_cb, 10)
        self.create_subscription(PointStamped, self.raw_tri_topic, self._raw_tri_cb, 10)

        # ── Timer ────────────────────────────────────────────────────────
        self.create_timer(1.0 / self.processing_rate_hz, self._timer_cb)

        self.get_logger().info(
            f'IMM overlay node started: image={self.image_topic}, '
            f'trajectory={self.predicted_path_topic}, output={self.output_topic}'
        )

    # ── Callbacks ────────────────────────────────────────────────────────

    def _image_cb(self, msg) -> None:
        self.latest_image = msg
        self._last_image_time = time.time()

    def _cam_info_cb(self, msg: CameraInfo) -> None:
        self.camera_info = msg

    def _path_cb(self, msg: Path) -> None:
        self.latest_path = msg

    def _estimate_cb(self, msg: Odometry) -> None:
        self.latest_estimate = msg

    def _raw_tri_cb(self, msg: PointStamped) -> None:
        self.latest_raw_tri = msg

    # ── Main processing loop ─────────────────────────────────────────────

    def _timer_cb(self) -> None:
        if self.latest_image is None or self.camera_info is None:
            return

        msg = self.latest_image

        # Decode image
        try:
            if self._use_compressed:
                bgr = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
            else:
                bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().error(f'Image decode failed: {exc}', throttle_duration_sec=5.0)
            return

        # Get camera intrinsics
        K = np.array(self.camera_info.k, dtype=np.float64).reshape(3, 3)
        camera_frame = self.camera_info.header.frame_id

        # Get world → camera transform
        try:
            now = self.get_clock().now()
            tf = self.tf_buffer.lookup_transform(
                camera_frame, self.world_frame_id,
                Time(), Duration(seconds=self.tf_timeout_sec),
            )
        except Exception as exc:
            self.get_logger().warn(f'TF lookup failed: {exc}', throttle_duration_sec=3.0)
            return

        t = tf.transform.translation
        q = tf.transform.rotation
        R_cw = quat_to_rotmat(q.x, q.y, q.z, q.w)
        t_cw = np.array([t.x, t.y, t.z], dtype=np.float64)

        overlay = bgr.copy()

        # ── Draw raw triangulation point ─────────────────────────────────
        if self.draw_raw_triangulation and self.latest_raw_tri is not None:
            p = self.latest_raw_tri.point
            pt_world = np.array([p.x, p.y, p.z], dtype=np.float64)
            px = self._project_point(pt_world, K, R_cw, t_cw, bgr.shape)
            if px is not None:
                cv2.drawMarker(overlay, px, self.raw_tri_color, cv2.MARKER_CROSS, 18, 2)
                cv2.putText(overlay, 'RAW', (px[0] + 10, px[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.raw_tri_color, 1, cv2.LINE_AA)

        # ── Draw current estimate ────────────────────────────────────────
        if self.latest_estimate is not None:
            ep = self.latest_estimate.pose.pose.position
            est_world = np.array([ep.x, ep.y, ep.z], dtype=np.float64)
            est_px = self._project_point(est_world, K, R_cw, t_cw, bgr.shape)
            if est_px is not None:
                cv2.circle(overlay, est_px, self.point_radius + 2, self.estimate_color, -1, cv2.LINE_AA)
                cv2.circle(overlay, est_px, self.point_radius + 4, self.estimate_color, 1, cv2.LINE_AA)
                cv2.putText(overlay, 'EST', (est_px[0] + 10, est_px[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.estimate_color, 1, cv2.LINE_AA)

                # Draw velocity arrow
                if self.draw_velocity_arrow:
                    vel = self.latest_estimate.twist.twist.linear
                    vel_vec = np.array([vel.x, vel.y, vel.z], dtype=np.float64)
                    speed = float(np.linalg.norm(vel_vec))
                    if speed > 0.1:
                        # Project a point 0.3s ahead along velocity
                        tip_world = est_world + vel_vec * 0.3
                        tip_px = self._project_point(tip_world, K, R_cw, t_cw, bgr.shape)
                        if tip_px is not None:
                            cv2.arrowedLine(overlay, est_px, tip_px, self.estimate_color, 2, cv2.LINE_AA, tipLength=0.3)

        # ── Draw predicted trajectory ────────────────────────────────────
        if self.latest_path is not None and len(self.latest_path.poses) > 0:
            projected_points: List[Optional[Tuple[int, int]]] = []
            n = len(self.latest_path.poses)

            for pose_stamped in self.latest_path.poses:
                p = pose_stamped.pose.position
                pt_world = np.array([p.x, p.y, p.z], dtype=np.float64)
                projected_points.append(self._project_point(pt_world, K, R_cw, t_cw, bgr.shape))

            # Draw connecting lines with color gradient
            for i in range(n - 1):
                if projected_points[i] is None or projected_points[i + 1] is None:
                    continue
                alpha = i / max(1, n - 2)
                color = self._lerp_color(self.traj_color_start, self.traj_color_end, alpha)
                cv2.line(overlay, projected_points[i], projected_points[i + 1],
                         color, self.line_thickness, cv2.LINE_AA)

            # Draw points with gradient
            for i, px in enumerate(projected_points):
                if px is None:
                    continue
                alpha = i / max(1, n - 1)
                color = self._lerp_color(self.traj_color_start, self.traj_color_end, alpha)
                radius = max(2, self.point_radius - int(alpha * 3))
                cv2.circle(overlay, px, radius, color, -1, cv2.LINE_AA)

            # Label first and last visible points
            if projected_points[0] is not None:
                cv2.putText(overlay, 'NOW', (projected_points[0][0] + 8, projected_points[0][1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.traj_color_start, 1, cv2.LINE_AA)
            for i in range(n - 1, -1, -1):
                if projected_points[i] is not None:
                    horizon_sec = len(self.latest_path.poses) * 0.04  # approximate
                    # Compute actual horizon from timestamps if available
                    if n >= 2:
                        t0 = self.latest_path.poses[0].header.stamp
                        t1 = self.latest_path.poses[-1].header.stamp
                        dt_ns = (t1.sec - t0.sec) * 1_000_000_000 + (t1.nanosec - t0.nanosec)
                        if dt_ns > 0:
                            horizon_sec = dt_ns / 1e9
                    cv2.putText(overlay, f'+{horizon_sec:.1f}s',
                                (projected_points[i][0] + 8, projected_points[i][1] - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.traj_color_end, 1, cv2.LINE_AA)
                    break

        # ── Draw legend ──────────────────────────────────────────────────
        self._draw_legend(overlay)

        # ── Publish ──────────────────────────────────────────────────────
        out_msg = CompressedImage()
        out_msg.header = msg.header
        out_msg.format = 'jpeg'
        out_msg.data = bytes(cv2.imencode('.jpg', overlay, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])[1])
        self.overlay_pub.publish(out_msg)

    # ── Helpers ──────────────────────────────────────────────────────────

    def _project_point(
        self,
        pt_world: np.ndarray,
        K: np.ndarray,
        R_cw: np.ndarray,
        t_cw: np.ndarray,
        img_shape: tuple,
    ) -> Optional[Tuple[int, int]]:
        """Project a 3D world point to pixel coordinates. Returns None if behind camera or out of frame."""
        pt_cam = R_cw @ pt_world + t_cw

        # Behind camera
        if pt_cam[2] <= 0.05:
            return None

        px_h = K @ pt_cam
        u = int(round(px_h[0] / px_h[2]))
        v = int(round(px_h[1] / px_h[2]))

        h, w = img_shape[:2]
        margin = 200  # allow slight off-screen for lines that cross the border
        if -margin <= u < w + margin and -margin <= v < h + margin:
            return (u, v)
        return None

    @staticmethod
    def _lerp_color(
        c0: Tuple[int, int, int],
        c1: Tuple[int, int, int],
        t: float,
    ) -> Tuple[int, int, int]:
        return (
            int(c0[0] + (c1[0] - c0[0]) * t),
            int(c0[1] + (c1[1] - c0[1]) * t),
            int(c0[2] + (c1[2] - c0[2]) * t),
        )

    def _draw_legend(self, img: np.ndarray) -> None:
        """Draw a small legend box in the top-left corner."""
        x0, y0 = 10, 10
        line_h = 18
        items = [
            (self.traj_color_start, 'IMM Prediction'),
            (self.estimate_color, 'Filtered Estimate'),
        ]
        if self.draw_raw_triangulation:
            items.append((self.raw_tri_color, 'Raw Triangulation'))

        # Background
        bw = 180
        bh = line_h * len(items) + 12
        sub = img[y0:y0 + bh, x0:x0 + bw]
        if sub.size > 0:
            img[y0:y0 + bh, x0:x0 + bw] = (sub * 0.4).astype(np.uint8)

        for i, (color, label) in enumerate(items):
            cy = y0 + 10 + i * line_h
            cv2.circle(img, (x0 + 10, cy), 5, color, -1, cv2.LINE_AA)
            cv2.putText(img, label, (x0 + 22, cy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)


def main() -> None:
    rclpy.init()
    node = IMMImageOverlayNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
