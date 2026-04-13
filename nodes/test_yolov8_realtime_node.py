#!/usr/bin/env python3

import os
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import rclpy
import yaml
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from ultralytics import YOLO


class TestYolov8RealtimeNode(Node):
    def __init__(self) -> None:
        super().__init__('test_yolov8_realtime_node')

        _weights_dir = str(Path(__file__).resolve().parent.parent / 'weights')
        default_model_path = f'{_weights_dir}/best_openvino_model'

        # Load observer name from runtime_stack.yaml
        script_dir = Path(__file__).resolve().parent
        config_path = script_dir.parent / 'config' / 'runtime_stack.yaml'
        observer_name = 'uav1'
        default_device = 'cpu'

        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    if config and 'uav_roles' in config:
                        observer_name = config['uav_roles'].get('observer1', 'uav1')
                    if config and 'vision' in config:
                        default_device = config['vision'].get('device', default_device)
                self.get_logger().info(f'Loaded config: observer1={observer_name}, device={default_device}')
            except Exception as e:
                self.get_logger().error(f'Failed to load config from {config_path}: {e}')
        else:
            self.get_logger().warning(f'Config file not found at {config_path}, defaulting to uav1')

        self.declare_parameter('input_image_topic', f'/{observer_name}/rgb/image_raw')
        self.declare_parameter('output_image_topic', '/test_img')
        self.declare_parameter(
            'model_path',
            default_model_path,
        )
        self.declare_parameter('confidence_threshold', 0.25)
        self.declare_parameter('iou_threshold', 0.45)
        self.declare_parameter('imgsz', 896)
        self.declare_parameter('max_detections', 100)
        self.declare_parameter('device', default_device)
        self.declare_parameter('line_width', 2)

        self.input_image_topic = str(self.get_parameter('input_image_topic').value)
        self.output_image_topic = str(self.get_parameter('output_image_topic').value)
        self.model_path = os.path.expanduser(str(self.get_parameter('model_path').value))
        self.confidence_threshold = float(self.get_parameter('confidence_threshold').value)
        self.iou_threshold = float(self.get_parameter('iou_threshold').value)
        self.imgsz = int(self.get_parameter('imgsz').value)
        self.max_detections = int(self.get_parameter('max_detections').value)
        self.device = str(self.get_parameter('device').value)
        self.line_width = max(1, int(self.get_parameter('line_width').value))

        default_model_path_expanded = os.path.expanduser(default_model_path)
        if not os.path.exists(self.model_path):
            if self.model_path == default_model_path_expanded:
                fallback_candidates = [
                    default_model_path_expanded,
                    os.path.expanduser(f'{_weights_dir}/best.onnx'),
                    os.path.expanduser(f'{_weights_dir}/best.pt'),
                ]
                for candidate in fallback_candidates:
                    if os.path.exists(candidate):
                        self.model_path = candidate
                        self.get_logger().warning(
                            f'Default model not found, using fallback: {self.model_path}'
                        )
                        break

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f'Model not found: {self.model_path}')

        self.bridge = CvBridge()
        self.model = YOLO(self.model_path)
        if self.model_path.endswith('.pt'):
            self.model.fuse()

        self.image_pub = self.create_publisher(Image, self.output_image_topic, 10)
        self.image_compressed_pub = self.create_publisher(CompressedImage, f'{self.output_image_topic}/compressed', 10)

        # Detect if input is compressed
        if self.input_image_topic.endswith('/compressed'):
            self.image_sub = self.create_subscription(
                CompressedImage,
                self.input_image_topic,
                self.image_callback,
                1,
            )
        else:
            self.image_sub = self.create_subscription(
                Image,
                self.input_image_topic,
                self.image_callback,
                1,
            )

        self.frame_counter = 0
        self.last_log_time = time.time()

        self.get_logger().info(f'Input image topic: {self.input_image_topic}')
        self.get_logger().info(f'Output image topic: {self.output_image_topic}')
        self.get_logger().info(f'YOLO model path: {self.model_path}')
        self.get_logger().info(f'Inference device: {self.device}')
        self.get_logger().info(f'Confidence threshold: {self.confidence_threshold:.2f}')
        self.get_logger().info(f'IOU threshold: {self.iou_threshold:.2f}')

    def image_callback(self, msg) -> None:
        try:
            if hasattr(msg, 'format'): # CompressedImage has 'format', Image has 'encoding'
                bgr_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
            else:
                bgr_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().error(f'Failed to convert input image: {exc}')
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

        annotated_image = self.draw_predictions(bgr_image, results)
        
        # Publish Raw
        out_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding='bgr8')
        out_msg.header = msg.header
        self.image_pub.publish(out_msg)

        # Publish Compressed (for WiFi visualization)
        out_msg_compressed = self.bridge.cv2_to_compressed_imgmsg(annotated_image, dst_format='jpg')
        out_msg_compressed.header = msg.header
        self.image_compressed_pub.publish(out_msg_compressed)

        self.frame_counter += 1
        now = time.time()
        if now - self.last_log_time >= 2.0:
            fps = self.frame_counter / (now - self.last_log_time)
            self.get_logger().info(f'Real-time test: FPS={fps:.2f}, last_inference={infer_ms:.1f}ms')
            self.frame_counter = 0
            self.last_log_time = now

    def draw_predictions(self, image: cv2.typing.MatLike, results) -> cv2.typing.MatLike:
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

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), self.line_width)
            y_text = y1 - 10 if y1 > 20 else y1 + 20
            cv2.putText(
                annotated,
                label,
                (x1, y_text),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        return annotated


def main() -> None:
    rclpy.init()
    node = TestYolov8RealtimeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()