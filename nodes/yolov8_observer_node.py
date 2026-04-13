#!/usr/bin/env python3

import os
import time
from dataclasses import dataclass
from pathlib import Path as FsPath
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, Vector3Stamped
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
        if _config_path.exists():
            import yaml
            try:
                with open(_config_path, 'r', encoding='utf-8') as _f:
                    _cfg = yaml.safe_load(_f)
                    if _cfg and 'vision' in _cfg:
                        _default_device = _cfg['vision'].get('device', _default_device)
                        _override_image_topic = _cfg['vision'].get('observer_image_topic', '') or ''
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
        self.declare_parameter('imgsz', 960)
        self.declare_parameter('max_detections', 100)
        self.declare_parameter('device', _default_device)
        self.declare_parameter('line_width', 2)
        self.declare_parameter('target_class_id', -1)
        self.declare_parameter('processing_rate_hz', 10.0)
        self.declare_parameter('tf_timeout_sec', 0.1)
        self.declare_parameter('world_frame_suffix', 'world_origin')
        self.declare_parameter('publish_annotated_image', True)

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
        
        # Timer
        self.create_timer(1.0 / self.processing_rate_hz, self.processing_timer_callback)

        self.get_logger().info(f'YOLO model path: {self.model_path}')
        self.get_logger().info(f'Observer ({self.uav_name}): {self.image_topic} -> {self.detection_topic}')
        self.get_logger().info(f'Publishing spatial info to: {self.camera_origin_topic} and {self.ray_topic}')
        self.get_logger().info(f'Processing rate: {self.processing_rate_hz:.1f} Hz')

    def camera_info_callback(self, msg: CameraInfo) -> None:
        self.camera_info = msg

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

        start_time = time.time()
        results = self.model.predict(
            source=bgr_image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            max_det=self.max_detections,
            device=self.device,
            verbose=False,
        )
        infer_ms = (time.time() - start_time) * 1000.0

        best_detection = self.extract_best_detection(results, msg)
        
        if self.publish_annotated_image:
            annotated_image = self.draw_predictions(bgr_image, results, best_detection)
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
            self.get_logger().info(f'Observer node: FPS={fps:.2f}, inference={infer_ms:.1f}ms, dropped={self._dropped_frames}')
            self.frame_counter = 0
            self._dropped_frames = 0
            self.last_log_time = now

    def extract_best_detection(self, results, msg: Image) -> Optional[DetectionObservation]:
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

        return max(candidates, key=lambda item: item.confidence)

    def draw_predictions(self, image: np.ndarray, results, best_detection: Optional[DetectionObservation]) -> np.ndarray:
        annotated = image.copy()

        if not results:
            return annotated

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return annotated

        names = result.names if result.names is not None else {}
        boxes_xyxy: List[Tuple[float, float, float, float]] = result.boxes.xyxy.tolist()
        confidences: List[float] = result.boxes.conf.tolist()
        classes: List[float] = result.boxes.cls.tolist()

        for box, conf, cls_idx in zip(boxes_xyxy, confidences, classes):
            x1, y1, x2, y2 = [int(v) for v in box]
            class_id = int(cls_idx)
            class_name = names.get(class_id, str(class_id)) if isinstance(names, dict) else str(class_id)
            label = f'{class_name} {conf:.2f}'

            color = (0, 255, 0)
            if best_detection is not None:
                center_u = 0.5 * (box[0] + box[2])
                center_v = 0.5 * (box[1] + box[3])
                if abs(center_u - best_detection.center_u) < 1.0 and abs(center_v - best_detection.center_v) < 1.0:
                    color = (0, 165, 255)
                    label += ' *'

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, self.line_width)
            y_text = y1 - 10 if y1 > 20 else y1 + 20
            cv2.putText(
                annotated,
                label,
                (x1, y_text),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )

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
            timeout = Duration(seconds=self.tf_timeout_sec)
            transform = self.tf_buffer.lookup_transform(
                self.world_frame_id, camera_frame, lookup_time, timeout
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

        # Build and publish origin
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
