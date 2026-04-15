#!/usr/bin/env python3

import os
import time
import collections
from dataclasses import dataclass
from pathlib import Path as FsPath
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, Vector3Stamped
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from tf2_ros import Buffer, TransformListener
from ultralytics import YOLO


def quat_to_rotmat(x: float, y: float, z: float, w: float) -> np.ndarray:
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def stamp_to_ns(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-9:
        raise ValueError('Cannot normalize near-zero vector')
    return vec / norm


@dataclass
class DetectionObservation:
    stamp_ns: int
    center_u: float
    center_v: float
    confidence: float
    class_id: int
    class_name: str


class Yolov8ObserverNode(Node):
    def __init__(self) -> None:
        super().__init__('yolov8_observer_node')

        _config_path = FsPath(__file__).resolve().parent.parent / 'config' / 'runtime_stack.yaml'
        _default_device = 'cuda'
        _override_image_topic = ''
        _default_publish_annotated = True
        _default_roi_enabled = False
        _default_roi_crop_imgsz = 320
        _default_roi_margin = 3.0
        _default_roi_min_size = 200
        _default_roi_timeout = 0.5
        _default_roi_full_interval = 10
        _default_roi_init_m = 3
        _default_roi_init_n = 5
        _default_roi_init_spread = 80
        _default_roi_sel_lambda = 0.3
        if _config_path.exists():
            import yaml
            try:
                with open(_config_path, 'r', encoding='utf-8') as _f:
                    _cfg = yaml.safe_load(_f)
                    if _cfg and 'vision' in _cfg:
                        _v = _cfg['vision']
                        _default_device = _v.get('device', _default_device)
                        _override_image_topic = _v.get('observer_image_topic', '') or ''
                        _default_publish_annotated = bool(_v.get('publish_annotated_image', _default_publish_annotated))
                        _default_roi_enabled = bool(_v.get('roi_crop_enabled', _default_roi_enabled))
                        _default_roi_crop_imgsz = int(_v.get('roi_crop_imgsz', _default_roi_crop_imgsz))
                        _default_roi_margin = float(_v.get('roi_margin_multiplier', _default_roi_margin))
                        _default_roi_min_size = int(_v.get('roi_min_size_px', _default_roi_min_size))
                        _default_roi_timeout = float(_v.get('roi_track_timeout_sec', _default_roi_timeout))
                        _default_roi_full_interval = int(_v.get('roi_full_scan_interval', _default_roi_full_interval))
                        _default_roi_init_m = int(_v.get('roi_init_m', _default_roi_init_m))
                        _default_roi_init_n = int(_v.get('roi_init_n', _default_roi_init_n))
                        _default_roi_init_spread = int(_v.get('roi_init_max_spread_px', _default_roi_init_spread))
                        _default_roi_sel_lambda = float(_v.get('roi_selection_lambda', _default_roi_sel_lambda))
            except Exception:
                pass

        _weights_dir = FsPath(__file__).resolve().parent.parent / 'weights'

        self.declare_parameter('uav_name', 'uav1')
        self.declare_parameter(
            'model_path',
            str(_weights_dir / 'best_openvino_model'),
        )
        self.declare_parameter('confidence_threshold', 0.25)
        self.declare_parameter('iou_threshold', 0.45)
        self.declare_parameter('imgsz', 896)
        self.declare_parameter('max_detections', 100)
        self.declare_parameter('device', _default_device)
        self.declare_parameter('line_width', 2)
        self.declare_parameter('target_class_id', -1)
        self.declare_parameter('processing_rate_hz', 100.0)
        self.declare_parameter('tf_timeout_sec', 0.05)
        self.declare_parameter('world_frame_suffix', 'world_origin')
        self.declare_parameter('publish_annotated_image', _default_publish_annotated)
        self.declare_parameter('roi_crop_enabled', _default_roi_enabled)
        self.declare_parameter('roi_crop_imgsz', _default_roi_crop_imgsz)
        self.declare_parameter('roi_margin_multiplier', _default_roi_margin)
        self.declare_parameter('roi_min_size_px', _default_roi_min_size)
        self.declare_parameter('roi_track_timeout_sec', _default_roi_timeout)
        self.declare_parameter('roi_full_scan_interval', _default_roi_full_interval)
        self.declare_parameter('roi_init_m', _default_roi_init_m)
        self.declare_parameter('roi_init_n', _default_roi_init_n)
        self.declare_parameter('roi_init_max_spread_px', _default_roi_init_spread)
        self.declare_parameter('roi_selection_lambda', _default_roi_sel_lambda)

        self.uav_name = str(self.get_parameter('uav_name').value)
        self.model_path = os.path.expanduser(str(self.get_parameter('model_path').value))
        self.confidence_threshold = float(self.get_parameter('confidence_threshold').value)
        self.iou_threshold = float(self.get_parameter('iou_threshold').value)
        self.imgsz = int(self.get_parameter('imgsz').value)
        self.max_detections = int(self.get_parameter('max_detections').value)
        self.device = str(self.get_parameter('device').value)
        self.line_width = max(1, int(self.get_parameter('line_width').value))
        self.target_class_id = int(self.get_parameter('target_class_id').value)
        self.processing_rate_hz = max(0.5, float(self.get_parameter('processing_rate_hz').value))
        self.tf_timeout_sec = float(self.get_parameter('tf_timeout_sec').value)
        self.world_frame_suffix = str(self.get_parameter('world_frame_suffix').value)
        self.publish_annotated_image = bool(self.get_parameter('publish_annotated_image').value)
        self.roi_crop_enabled = bool(self.get_parameter('roi_crop_enabled').value)
        self.roi_crop_imgsz = int(self.get_parameter('roi_crop_imgsz').value)
        self.roi_margin = float(self.get_parameter('roi_margin_multiplier').value)
        self.roi_min_size = int(self.get_parameter('roi_min_size_px').value)
        self.roi_timeout_ns = int(float(self.get_parameter('roi_track_timeout_sec').value) * 1_000_000_000)
        self.roi_full_scan_interval = max(1, int(self.get_parameter('roi_full_scan_interval').value))
        self.roi_init_m = max(1, int(self.get_parameter('roi_init_m').value))
        self.roi_init_n = max(self.roi_init_m, int(self.get_parameter('roi_init_n').value))
        self.roi_init_spread = float(self.get_parameter('roi_init_max_spread_px').value)
        self.roi_sel_lambda = float(self.get_parameter('roi_selection_lambda').value)

        self.image_topic = _override_image_topic or f'/{self.uav_name}/rgb/image_raw'
        self._use_compressed = self.image_topic.endswith('/compressed')
        _img_base = self.image_topic.replace('/compressed', '').rsplit('/image_raw', 1)[0]
        self.camera_info_topic = f'{_img_base}/camera_info'
        self.detection_topic = f'/{self.uav_name}/detection'
        self.camera_origin_topic = f'/{self.uav_name}/camera_origin'
        self.ray_topic = f'/{self.uav_name}/ray'

        # Auto-derive world frame as in the original node
        self.world_frame_id = f'{self.uav_name}/{self.world_frame_suffix}'

        if not os.path.exists(self.model_path):
            _fallback = str(_weights_dir / 'best.pt')
            if os.path.exists(_fallback):
                self.get_logger().warn(f'Model not found at {self.model_path}, falling back to {_fallback}')
                self.model_path = _fallback
            else:
                raise FileNotFoundError(f'Model not found: {self.model_path} (fallback {_fallback} also missing)')

        self.bridge = CvBridge()
        self.model = YOLO(self.model_path, task='detect')
        if self.model_path.endswith('.pt'):
            self.model.fuse()

        self.camera_info: Optional[CameraInfo] = None
        self.latest_image = None
        self.last_processed_image_stamp_ns: int = -1

        self.frame_counter = 0
        self._dropped_frames = 0
        self.last_log_time = time.time()
        self._last_image_time = time.time()

        # ROI tracker state (2D pixel-space)
        self._roi_last_bbox: Optional[Tuple[float, float, float, float]] = None  # x1,y1,x2,y2
        self._roi_last_center: Optional[Tuple[float, float]] = None  # u,v
        self._roi_last_stamp_ns: int = 0
        self._roi_frame_counter: int = 0  # counts since last full scan
        self._roi_vel_u: float = 0.0  # pixel velocity estimate
        self._roi_vel_v: float = 0.0
        self._roi_consecutive_misses: int = 0  # ROI escalation counter
        self._init_gate_buffer: collections.deque = collections.deque(maxlen=self.roi_init_n)  # M-of-N init gate

        # IMM odometry from main node (for ROI center prediction)
        self._target_odom: Optional[Odometry] = None
        self._target_odom_stamp_ns: int = 0

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Publishers
        _det_type = CompressedImage if self.publish_annotated_image else Image
        self.detection_pub = self.create_publisher(CompressedImage, self.detection_topic + '/compressed', 10)
        self.origin_pub = self.create_publisher(PointStamped, self.camera_origin_topic, 10)
        self.ray_pub = self.create_publisher(Vector3Stamped, self.ray_topic, 10)

        # Subscribers — BEST_EFFORT for real-time sensor data over WiFi
        _sensor_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        if self._use_compressed:
            self.create_subscription(CompressedImage, self.image_topic, self.image_callback, _sensor_qos)
        else:
            self.create_subscription(Image, self.image_topic, self.image_callback, _sensor_qos)
        self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_callback, 10)
        self.create_subscription(Odometry, '/target/odometry_estimate', self._target_odom_callback, 10)

        # Timer
        self.create_timer(1.0 / self.processing_rate_hz, self.processing_timer_callback)

        self.get_logger().info(f'YOLO model path: {self.model_path}')
        self.get_logger().info(f'Observer ({self.uav_name}): {self.image_topic} -> {self.detection_topic}')
        self.get_logger().info(f'Publishing spatial info to: {self.camera_origin_topic} and {self.ray_topic}')
        self.get_logger().info(f'Processing rate: {self.processing_rate_hz:.1f} Hz')

    def camera_info_callback(self, msg: CameraInfo) -> None:
        self.camera_info = msg

    def _target_odom_callback(self, msg: Odometry) -> None:
        self._target_odom = msg
        self._target_odom_stamp_ns = stamp_to_ns(msg.header.stamp)

    def _get_roi_center(self, stamp_ns: int, img_w: int, img_h: int) -> Optional[Tuple[float, float]]:
        """Get ROI center: IMM 3D→2D projection (from main node odometry), fallback to 2D pixel velocity."""
        # Strategy 1: project IMM 3D position from /target/odometry_estimate into image plane
        if self._target_odom is not None and self.camera_info is not None:
            odom_age_ns = abs(stamp_ns - self._target_odom_stamp_ns)
            if odom_age_ns < 2_000_000_000:  # odometry not older than 2 s
                camera_frame = self.camera_info.header.frame_id
                try:
                    transform = self.tf_buffer.lookup_transform(
                        camera_frame, self._target_odom.header.frame_id, Time(), Duration(seconds=0)
                    )
                    t = transform.transform.translation
                    q = transform.transform.rotation
                    r_cw = quat_to_rotmat(q.x, q.y, q.z, q.w)
                    t_cw = np.array([t.x, t.y, t.z], dtype=np.float64)

                    pos_3d = np.array([
                        self._target_odom.pose.pose.position.x,
                        self._target_odom.pose.pose.position.y,
                        self._target_odom.pose.pose.position.z,
                    ], dtype=np.float64)
                    vel_3d = np.array([
                        self._target_odom.twist.twist.linear.x,
                        self._target_odom.twist.twist.linear.y,
                        self._target_odom.twist.twist.linear.z,
                    ], dtype=np.float64)
                    dt = (stamp_ns - self._target_odom_stamp_ns) / 1_000_000_000.0
                    pred_3d = pos_3d + vel_3d * dt

                    p_cam = r_cw @ pred_3d + t_cw
                    if p_cam[2] > 0.1:
                        fx = float(self.camera_info.k[0])
                        fy = float(self.camera_info.k[4])
                        cx = float(self.camera_info.k[2])
                        cy = float(self.camera_info.k[5])
                        u = fx * p_cam[0] / p_cam[2] + cx
                        v = fy * p_cam[1] / p_cam[2] + cy
                        if 0 <= u < img_w and 0 <= v < img_h:
                            return (u, v)
                except Exception:
                    pass

        # Strategy 2: 2D pixel velocity tracker
        if self._roi_last_center is not None and (stamp_ns - self._roi_last_stamp_ns) < self.roi_timeout_ns:
            dt_sec = (stamp_ns - self._roi_last_stamp_ns) / 1_000_000_000.0
            pred_u = self._roi_last_center[0] + self._roi_vel_u * dt_sec
            pred_v = self._roi_last_center[1] + self._roi_vel_v * dt_sec
            return (pred_u, pred_v)

        return None

    def _check_init_gate(self, center_u: float, center_v: float, stamp_ns: int) -> bool:
        """M-of-N init gate: require M consistent detections out of N before starting track."""
        self._init_gate_buffer.append((center_u, center_v, stamp_ns))
        if len(self._init_gate_buffer) < self.roi_init_m:
            return False
        centers = [(u, v) for u, v, _ in self._init_gate_buffer]
        # Check if at least M of the last N centers cluster within roi_init_spread
        for i in range(len(centers)):
            ref_u, ref_v = centers[i]
            close = sum(1 for u, v in centers if abs(u - ref_u) < self.roi_init_spread and abs(v - ref_v) < self.roi_init_spread)
            if close >= self.roi_init_m:
                return True
        return False

    def image_callback(self, msg) -> None:
        if self.latest_image is not None:
            self._dropped_frames += 1
        self.latest_image = msg
        self._last_image_time = time.time()

    def processing_timer_callback(self) -> None:
        msg = self.latest_image
        if msg is None or self.camera_info is None:
            if time.time() - self._last_image_time > 3.0:
                self.get_logger().warn('No image received for 3 s!', throttle_duration_sec=5.0)
            return

        stamp_ns = stamp_to_ns(msg.header.stamp)
        if stamp_ns <= self.last_processed_image_stamp_ns:
            return

        self.last_processed_image_stamp_ns = stamp_ns
        self.process_image(msg)

    def process_image(self, msg) -> None:
        try:
            if self._use_compressed:
                bgr_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
            else:
                bgr_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().error(f'[{self.uav_name}] Failed to convert input image: {exc}')
            return

        stamp_ns = stamp_to_ns(msg.header.stamp)
        img_h, img_w = bgr_image.shape[:2]

        # Decide: ROI crop or full scan
        use_roi = False
        roi_x1 = roi_y1 = roi_x2 = roi_y2 = 0
        infer_image = bgr_image
        infer_imgsz = self.imgsz

        if self.roi_crop_enabled and self._roi_last_bbox is not None:
            force_full = (self._roi_frame_counter % self.roi_full_scan_interval == 0)

            roi_center = self._get_roi_center(stamp_ns, img_w, img_h)
            has_track = roi_center is not None

            if has_track and not force_full:
                pred_u, pred_v = roi_center

                # ROI size from last bbox — landscape (match image aspect ratio)
                bx1, by1, bx2, by2 = self._roi_last_bbox
                bbox_w = bx2 - bx1
                bbox_h = by2 - by1
                roi_h = max(bbox_w, bbox_h) * self.roi_margin
                roi_h = max(roi_h, float(self.roi_min_size))

                # Escalation: expand ROI on consecutive misses
                if self._roi_consecutive_misses >= 3:
                    roi_h *= 2.0
                elif self._roi_consecutive_misses >= 1:
                    roi_h *= 1.5

                aspect = img_w / img_h  # e.g. 1920/1080 ≈ 1.778
                roi_w = roi_h * aspect

                roi_x1 = int(max(0.0, pred_u - roi_w * 0.5))
                roi_y1 = int(max(0.0, pred_v - roi_h * 0.5))
                roi_x2 = int(min(float(img_w), pred_u + roi_w * 0.5))
                roi_y2 = int(min(float(img_h), pred_v + roi_h * 0.5))

                if (roi_x2 - roi_x1) >= self.roi_min_size and (roi_y2 - roi_y1) >= self.roi_min_size:
                    infer_image = bgr_image[roi_y1:roi_y2, roi_x1:roi_x2]
                    infer_imgsz = self.roi_crop_imgsz
                    use_roi = True

        self._roi_frame_counter += 1

        start_time = time.time()
        results = self.model.predict(
            source=infer_image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=infer_imgsz,
            max_det=self.max_detections,
            device=self.device,
            verbose=False,
        )
        infer_ms = (time.time() - start_time) * 1000.0

        # Pass ROI center for prediction-gated selection
        roi_pred_center = None
        if use_roi and self._roi_last_center is not None:
            roi_pred_center = self._roi_last_center
        best_detection = self.extract_best_detection(results, msg, roi_pred_center, roi_x1, roi_y1,
                                                      roi_x2 - roi_x1 if use_roi else img_w)

        # Remap detection coordinates from crop to full image
        if best_detection is not None and use_roi:
            best_detection.center_u += roi_x1
            best_detection.center_v += roi_y1

        # Update ROI tracker
        if best_detection is not None:
            if self._roi_last_bbox is not None:
                # Active track — update tracker, reset misses
                self._update_roi_tracker(best_detection, results, stamp_ns, roi_x1, roi_y1)
                self._roi_consecutive_misses = 0
            else:
                # No active track — apply M-of-N init gate
                if self._check_init_gate(best_detection.center_u, best_detection.center_v, stamp_ns):
                    self._update_roi_tracker(best_detection, results, stamp_ns, roi_x1, roi_y1)
                    self._roi_consecutive_misses = 0
                    self._init_gate_buffer.clear()
                    self.get_logger().info('ROI track initialized via M-of-N gate')
        elif self.roi_crop_enabled and self._roi_last_bbox is not None:
            self._roi_consecutive_misses += 1
            if (stamp_ns - self._roi_last_stamp_ns) >= self.roi_timeout_ns:
                self._roi_last_bbox = None
                self._roi_last_center = None
                self._roi_consecutive_misses = 0

        if self.publish_annotated_image:
            annotated_image = self.draw_predictions(bgr_image, results, best_detection, use_roi, roi_x1, roi_y1, roi_x2, roi_y2)
            comp_msg = CompressedImage()
            comp_msg.header = msg.header
            comp_msg.format = 'jpeg'
            comp_msg.data = bytes(cv2.imencode('.jpg', annotated_image, [cv2.IMWRITE_JPEG_QUALITY, 75])[1])
            self.detection_pub.publish(comp_msg)

        if best_detection is not None:
            self.compute_and_publish_ray(best_detection)

        self.frame_counter += 1
        now = time.time()
        if now - self.last_log_time >= 2.0:
            fps = self.frame_counter / (now - self.last_log_time)
            mode = 'ROI' if use_roi else 'FULL'
            esc = f' esc={self._roi_consecutive_misses}' if self._roi_consecutive_misses > 0 else ''
            self.get_logger().info(f'Observer node: FPS={fps:.2f}, inference={infer_ms:.1f}ms, mode={mode}{esc}, dropped={self._dropped_frames}')
            self.frame_counter = 0
            self._dropped_frames = 0
            self.last_log_time = now

    def _update_roi_tracker(self, det: 'DetectionObservation', results, stamp_ns: int, roi_x1: int, roi_y1: int) -> None:
        """Update 2D ROI tracker from detection + bbox."""
        # Extract bbox for the best detection from results
        result = results[0]
        boxes_xyxy = result.boxes.xyxy.tolist()
        confs = result.boxes.conf.tolist()
        best_idx = int(np.argmax(confs))
        bx1, by1, bx2, by2 = boxes_xyxy[best_idx]
        # Remap bbox to full image coords
        bx1 += roi_x1; by1 += roi_y1
        bx2 += roi_x1; by2 += roi_y1

        new_u = det.center_u
        new_v = det.center_v

        # Estimate pixel velocity (simple exponential smoothing)
        if self._roi_last_center is not None and self._roi_last_stamp_ns > 0:
            dt = (stamp_ns - self._roi_last_stamp_ns) / 1_000_000_000.0
            if dt > 0.001:
                alpha = 0.6  # smoothing factor
                raw_vel_u = (new_u - self._roi_last_center[0]) / dt
                raw_vel_v = (new_v - self._roi_last_center[1]) / dt
                self._roi_vel_u = alpha * raw_vel_u + (1.0 - alpha) * self._roi_vel_u
                self._roi_vel_v = alpha * raw_vel_v + (1.0 - alpha) * self._roi_vel_v

        self._roi_last_bbox = (bx1, by1, bx2, by2)
        self._roi_last_center = (new_u, new_v)
        self._roi_last_stamp_ns = stamp_ns

    def extract_best_detection(self, results, msg: Image, roi_pred_center: Optional[Tuple[float, float]] = None,
                               roi_x1: int = 0, roi_y1: int = 0, roi_half_side: float = 1.0) -> Optional[DetectionObservation]:
        if not results:
            return None

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return None

        names = result.names if result.names is not None else {}
        boxes_xyxy: List[Tuple[float, float, float, float]] = result.boxes.xyxy.tolist()
        confidences: List[float] = result.boxes.conf.tolist()
        classes: List[float] = result.boxes.cls.tolist()

        candidates: List[DetectionObservation] = []
        for box, conf, cls_idx in zip(boxes_xyxy, confidences, classes):
            class_id = int(cls_idx)
            if self.target_class_id >= 0 and class_id != self.target_class_id:
                continue

            x1, y1, x2, y2 = box
            center_u = 0.5 * (x1 + x2)
            center_v = 0.5 * (y1 + y2)
            class_name = names.get(class_id, str(class_id)) if isinstance(names, dict) else str(class_id)
            candidates.append(
                DetectionObservation(
                    stamp_ns=stamp_to_ns(msg.header.stamp),
                    center_u=center_u,
                    center_v=center_v,
                    confidence=float(conf),
                    class_id=class_id,
                    class_name=class_name,
                )
            )

        if not candidates:
            return None

        # Prediction-gated selection: when ROI track is active, prefer detections near prediction
        if roi_pred_center is not None and self.roi_sel_lambda > 0.0:
            pred_u, pred_v = roi_pred_center
            d_max = max(roi_half_side * 0.5, 1.0)
            def score(det):
                # Coordinates are still in crop space here — offset by roi_x1/y1 for comparison
                d_pixel = ((det.center_u + roi_x1 - pred_u) ** 2 + (det.center_v + roi_y1 - pred_v) ** 2) ** 0.5
                return det.confidence - self.roi_sel_lambda * min(d_pixel / d_max, 1.0)
            return max(candidates, key=score)

        return max(candidates, key=lambda item: item.confidence)

    def draw_predictions(self, image: np.ndarray, results, best_detection: Optional[DetectionObservation], use_roi: bool = False, roi_x1: int = 0, roi_y1: int = 0, roi_x2: int = 0, roi_y2: int = 0) -> np.ndarray:
        annotated = image.copy()

        if not results:
            return annotated

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            # Still draw ROI region even if no detections
            if use_roi and roi_x2 > roi_x1 and roi_y2 > roi_y1:
                cv2.rectangle(annotated, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 220, 0), 2)
                cv2.putText(annotated, 'ROI', (roi_x1, roi_y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1, cv2.LINE_AA)
            return annotated

        names = result.names if result.names is not None else {}
        boxes_xyxy: List[Tuple[float, float, float, float]] = result.boxes.xyxy.tolist()
        confidences: List[float] = result.boxes.conf.tolist()
        classes: List[float] = result.boxes.cls.tolist()

        for box, conf, cls_idx in zip(boxes_xyxy, confidences, classes):
            # Remap from crop to full image
            x1 = int(box[0]) + roi_x1
            y1 = int(box[1]) + roi_y1
            x2 = int(box[2]) + roi_x1
            y2 = int(box[3]) + roi_y1
            class_id = int(cls_idx)
            class_name = names.get(class_id, str(class_id)) if isinstance(names, dict) else str(class_id)
            label = f'{class_name} {conf:.2f}'

            color = (0, 255, 0)
            if best_detection is not None:
                center_u = 0.5 * (box[0] + box[2]) + roi_x1
                center_v = 0.5 * (box[1] + box[3]) + roi_y1
                if abs(center_u - best_detection.center_u) < 1.0 and abs(center_v - best_detection.center_v) < 1.0:
                    color = (0, 165, 255)
                    label += ' *'

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, self.line_width)
            y_text = y1 - 10 if y1 > 20 else y1 + 20
            cv2.putText(annotated, label, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        # Draw actual ROI crop region (green rectangle with label)
        if use_roi and roi_x2 > roi_x1 and roi_y2 > roi_y1:
            cv2.rectangle(annotated, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 220, 0), 2)
            cv2.putText(annotated, 'ROI', (roi_x1, roi_y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1, cv2.LINE_AA)

        return annotated

    def compute_and_publish_ray(self, obs: DetectionObservation) -> None:
        if self.camera_info is None:
            return

        fx = float(self.camera_info.k[0])
        fy = float(self.camera_info.k[4])
        cx = float(self.camera_info.k[2])
        cy = float(self.camera_info.k[5])

        ray_optical = normalize(np.array([(obs.center_u - cx) / fx, (obs.center_v - cy) / fy, 1.0], dtype=np.float64))

        camera_frame = self.camera_info.header.frame_id
        # The world_frame_id handles the prefix internally since we formed it as {uav_name}/{suffix}

        try:
            sec = obs.stamp_ns // 1_000_000_000
            nanosec = obs.stamp_ns % 1_000_000_000
            lookup_time = Time(seconds=sec, nanoseconds=nanosec)
            transform = self.tf_buffer.lookup_transform(
                self.world_frame_id, camera_frame, lookup_time, Duration(seconds=0),
            )
        except Exception:
            # Exact timestamp not available yet — use latest TF (small pose error, never blocks)
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.world_frame_id, camera_frame, Time(), Duration(seconds=0),
                )
            except Exception as exc:
                self.get_logger().warn(
                    f'TF lookup {self.world_frame_id} <- {camera_frame} failed: {exc}',
                    throttle_duration_sec=2.0,
                )
                return

        t = transform.transform.translation
        q = transform.transform.rotation
        camera_origin_world = np.array([t.x, t.y, t.z], dtype=np.float64)
        r_wc = quat_to_rotmat(q.x, q.y, q.z, q.w)
        ray_world = normalize(r_wc @ ray_optical)

        # Build and publish origin (use observation timestamp for message sync)
        sec = obs.stamp_ns // 1_000_000_000
        nanosec = obs.stamp_ns % 1_000_000_000
        origin_msg = PointStamped()
        origin_msg.header.frame_id = self.world_frame_id
        origin_msg.header.stamp.sec = sec
        origin_msg.header.stamp.nanosec = nanosec
        origin_msg.point.x = float(camera_origin_world[0])
        origin_msg.point.y = float(camera_origin_world[1])
        origin_msg.point.z = float(camera_origin_world[2])
        self.origin_pub.publish(origin_msg)

        # Build and publish ray
        ray_msg = Vector3Stamped()
        ray_msg.header.frame_id = self.world_frame_id
        ray_msg.header.stamp.sec = sec
        ray_msg.header.stamp.nanosec = nanosec
        ray_msg.vector.x = float(ray_world[0])
        ray_msg.vector.y = float(ray_world[1])
        ray_msg.vector.z = float(ray_world[2])
        self.ray_pub.publish(ray_msg)


def main() -> None:
    rclpy.init()
    node = Yolov8ObserverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
