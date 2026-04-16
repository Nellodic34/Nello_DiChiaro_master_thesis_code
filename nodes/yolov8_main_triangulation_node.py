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
import message_filters
from geometry_msgs.msg import Point, PointStamped, PoseStamped, Vector3Stamped
from nav_msgs.msg import Odometry, Path
from visualization_msgs.msg import Marker, MarkerArray
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from tf2_ros import Buffer, TransformListener, StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped
import math
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


def triangulate_midpoint(
    origin_a: np.ndarray,
    dir_a: np.ndarray,
    origin_b: np.ndarray,
    dir_b: np.ndarray,
) -> Tuple[np.ndarray, float]:
    w0 = origin_a - origin_b
    a = float(np.dot(dir_a, dir_a))
    b = float(np.dot(dir_a, dir_b))
    c = float(np.dot(dir_b, dir_b))
    d = float(np.dot(dir_a, w0))
    e = float(np.dot(dir_b, w0))
    denom = a * c - b * b

    if abs(denom) < 1e-9:
        midpoint = 0.5 * (origin_a + origin_b)
        return midpoint, float(np.linalg.norm(origin_a - origin_b))

    s = (b * e - c * d) / denom
    t = (a * e - b * d) / denom

    closest_a = origin_a + s * dir_a
    closest_b = origin_b + t * dir_b
    midpoint = 0.5 * (closest_a + closest_b)
    error = float(np.linalg.norm(closest_a - closest_b))
    return midpoint, error


@dataclass
class StateCov:
    x: np.ndarray
    P: np.ndarray


class LinearKalmanFilter:
    def __init__(self, A: np.ndarray, B: np.ndarray, H: np.ndarray) -> None:
        self.A = A.copy()
        self.B = B.copy()
        self.H = H.copy()
        self.n = A.shape[0]

    def predict(self, sc: StateCov, u: np.ndarray, Q: np.ndarray, dt: float) -> StateCov:
        x_pred = self.A @ sc.x + self.B @ u
        P_pred = self.A @ sc.P @ self.A.T + Q
        return StateCov(x=x_pred, P=P_pred)

    def correct(self, sc: StateCov, z: np.ndarray, R: np.ndarray) -> StateCov:
        y = z - self.H @ sc.x
        S = self.H @ sc.P @ self.H.T + R
        K = sc.P @ self.H.T @ np.linalg.inv(S)
        x_new = sc.x + K @ y
        P_new = (np.eye(self.n) - K @ self.H) @ sc.P
        return StateCov(x=x_new, P=P_new)


def build_transition_matrix_ca(dt: float) -> np.ndarray:
    A = np.eye(9, dtype=np.float64)
    dt2 = 0.5 * dt * dt
    A[0, 3] = dt;  A[0, 6] = dt2
    A[1, 4] = dt;  A[1, 7] = dt2
    A[2, 5] = dt;  A[2, 8] = dt2
    A[3, 6] = dt
    A[4, 7] = dt
    A[5, 8] = dt
    return A


def build_process_noise_ca(dt: float, sigma_a: float) -> np.ndarray:
    dt = float(max(0.0, dt))
    q = float(max(1e-9, sigma_a) ** 2)

    dt2 = dt * dt
    dt3 = dt2 * dt
    dt4 = dt3 * dt
    dt5 = dt4 * dt

    q_block = np.array(
        [
            [dt5 / 20.0, dt4 / 8.0, dt3 / 6.0],
            [dt4 / 8.0, dt3 / 3.0, dt2 / 2.0],
            [dt3 / 6.0, dt2 / 2.0, dt],
        ],
        dtype=np.float64,
    )
    q_block *= q

    Q = np.zeros((9, 9), dtype=np.float64)
    for axis in range(3):
        idx = [axis, axis + 3, axis + 6]
        Q[np.ix_(idx, idx)] = q_block
    return Q


def build_transition_matrix_cv(dt: float) -> np.ndarray:
    A = np.eye(6, dtype=np.float64)
    A[0, 3] = dt
    A[1, 4] = dt
    A[2, 5] = dt
    return A


def build_process_noise_cv(dt: float, sigma_v: float) -> np.ndarray:
    dt = float(max(0.0, dt))
    q = float(max(1e-9, sigma_v) ** 2)

    dt2 = dt * dt
    dt3 = dt2 * dt

    q_block = np.array(
        [
            [dt3 / 3.0, dt2 / 2.0],
            [dt2 / 2.0, dt],
        ],
        dtype=np.float64,
    )
    q_block *= q

    Q = np.zeros((6, 6), dtype=np.float64)
    for axis in range(3):
        idx = [axis, axis + 3]
        Q[np.ix_(idx, idx)] = q_block
    return Q


class IMMFilter:
    def __init__(self, pi: np.ndarray, model_probs: np.ndarray):
        self.pi = pi.copy()
        self.mu = model_probs.copy()
        
        A0 = np.eye(6, dtype=np.float64)
        B0 = np.zeros((6, 1), dtype=np.float64)
        H0 = np.zeros((3, 6), dtype=np.float64)
        H0[0, 0] = H0[1, 1] = H0[2, 2] = 1.0
        self.kf_cv = LinearKalmanFilter(A0, B0, H0)
        self.sc_cv: Optional[StateCov] = None
        
        A1 = np.eye(9, dtype=np.float64)
        B1 = np.zeros((9, 1), dtype=np.float64)
        H1 = np.zeros((3, 9), dtype=np.float64)
        H1[0, 0] = H1[1, 1] = H1[2, 2] = 1.0
        self.kf_ca = LinearKalmanFilter(A1, B1, H1)
        self.sc_ca: Optional[StateCov] = None
        
        self.n = 2

    def initialize(self, measurement: np.ndarray, P_init_diag: float):
        x0_cv = np.zeros(6, dtype=np.float64)
        x0_cv[:3] = measurement
        P0_cv = np.eye(6, dtype=np.float64) * P_init_diag
        self.sc_cv = StateCov(x=x0_cv, P=P0_cv)
        
        x0_ca = np.zeros(9, dtype=np.float64)
        x0_ca[:3] = measurement
        P0_ca = np.eye(9, dtype=np.float64) * P_init_diag
        self.sc_ca = StateCov(x=x0_ca, P=P0_ca)

    def _state_ca_to_cv(self, x_ca: np.ndarray) -> np.ndarray:
        return x_ca[:6]
    
    def _cov_ca_to_cv(self, P_ca: np.ndarray) -> np.ndarray:
        return P_ca[:6, :6]
        
    def _state_cv_to_ca(self, x_cv: np.ndarray) -> np.ndarray:
        x_ca = np.zeros(9, dtype=np.float64)
        x_ca[:6] = x_cv
        return x_ca
        
    def _cov_cv_to_ca(self, P_cv: np.ndarray) -> np.ndarray:
        P_ca = np.zeros((9, 9), dtype=np.float64)
        P_ca[:6, :6] = P_cv
        P_ca[6, 6] = P_ca[7, 7] = P_ca[8, 8] = 1.0 
        return P_ca

    def predict_and_correct(self, dt: float, Q_cv: np.ndarray, Q_ca: np.ndarray, z: np.ndarray, R: np.ndarray, mahalanobis_thresh: float = 9.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        assert self.sc_cv is not None
        assert self.sc_ca is not None
        
        c_bar = self.pi.T @ self.mu
        mu_mix = np.zeros((2, 2), dtype=np.float64)
        for i in range(2):
            for j in range(2):
                mu_mix[i, j] = self.pi[i, j] * self.mu[i] / max(c_bar[j], 1e-12)
        
        x_cv = self.sc_cv.x
        x_ca = self.sc_ca.x
        P_cv = self.sc_cv.P
        P_ca = self.sc_ca.P
        
        x_ca_as_cv = self._state_ca_to_cv(x_ca)
        x_cv_as_ca = self._state_cv_to_ca(x_cv)
        x_cv_as_ca[6:9] = x_ca[6:9]
        
        x0_mix_cv = x_cv * mu_mix[0, 0] + x_ca_as_cv * mu_mix[1, 0]
        x0_mix_ca = x_cv_as_ca * mu_mix[0, 1] + x_ca * mu_mix[1, 1]
        
        dx0 = x_cv - x0_mix_cv
        dx1 = x_ca_as_cv - x0_mix_cv
        P0_mix_cv = mu_mix[0, 0] * (P_cv + np.outer(dx0, dx0)) + mu_mix[1, 0] * (self._cov_ca_to_cv(P_ca) + np.outer(dx1, dx1))
        
        dx0_ca = x_cv_as_ca - x0_mix_ca
        dx1_ca = x_ca - x0_mix_ca
        P0_mix_ca = mu_mix[0, 1] * (self._cov_cv_to_ca(P_cv) + np.outer(dx0_ca, dx0_ca)) + mu_mix[1, 1] * (P_ca + np.outer(dx1_ca, dx1_ca))
        
        self.sc_cv.x = x0_mix_cv
        self.sc_cv.P = P0_mix_cv
        self.sc_ca.x = x0_mix_ca
        self.sc_ca.P = P0_mix_ca
        
        u_cv = np.zeros(1, dtype=np.float64)
        u_ca = np.zeros(1, dtype=np.float64)
        
        self.kf_cv.A = build_transition_matrix_cv(dt)
        sc_pred_cv = self.kf_cv.predict(self.sc_cv, u_cv, Q_cv, dt)
        
        self.kf_ca.A = build_transition_matrix_ca(dt)
        sc_pred_ca = self.kf_ca.predict(self.sc_ca, u_ca, Q_ca, dt)
        
        y_cv = z - self.kf_cv.H @ sc_pred_cv.x
        S_cv = self.kf_cv.H @ sc_pred_cv.P @ self.kf_cv.H.T + R
        
        y_ca = z - self.kf_ca.H @ sc_pred_ca.x
        S_ca = self.kf_ca.H @ sc_pred_ca.P @ self.kf_ca.H.T + R
        
        try:
            invS_cv_true = np.linalg.inv(S_cv)
            d_cv = float(y_cv.T @ invS_cv_true @ y_cv)
        except np.linalg.LinAlgError:
            d_cv = float('inf')
        
        try:
            invS_ca_true = np.linalg.inv(S_ca)
            d_ca = float(y_ca.T @ invS_ca_true @ y_ca)
        except np.linalg.LinAlgError:
            d_ca = float('inf')
        
        accepted = True
        if d_cv > mahalanobis_thresh and d_ca > mahalanobis_thresh:
            self.sc_cv = sc_pred_cv
            self.sc_ca = sc_pred_ca
            accepted = False
            
            x_comb = self._state_cv_to_ca(self.sc_cv.x) * self.mu[0] + self.sc_ca.x * self.mu[1]
            dx0_comb = self._state_cv_to_ca(self.sc_cv.x) - x_comb
            dx1_comb = self.sc_ca.x - x_comb
            P_comb = self.mu[0] * (self._cov_cv_to_ca(self.sc_cv.P) + np.outer(dx0_comb, dx0_comb)) + \
                     self.mu[1] * (self.sc_ca.P + np.outer(dx1_comb, dx1_comb))
            return x_comb, P_comb, self.mu, accepted
        
        def mv_normal_pdf_fixed(y, S_fixed):
            try:
                invS = np.linalg.inv(S_fixed)
            except np.linalg.LinAlgError:
                return 1e-12
            exponent = -0.5 * y.T @ invS @ y
            return float(np.exp(max(-50.0, min(50.0, exponent))))
            
        Lambda = np.array([mv_normal_pdf_fixed(y_cv, R), mv_normal_pdf_fixed(y_ca, R)])
        
        self.sc_cv = self.kf_cv.correct(sc_pred_cv, z, R)
        self.sc_ca = self.kf_ca.correct(sc_pred_ca, z, R)
        
        self.mu = Lambda * c_bar
        self.mu /= np.sum(self.mu) + 1e-12
        
        x_comb = self._state_cv_to_ca(self.sc_cv.x) * self.mu[0] + self.sc_ca.x * self.mu[1]
        
        dx0_comb = self._state_cv_to_ca(self.sc_cv.x) - x_comb
        dx1_comb = self.sc_ca.x - x_comb
        
        P_comb = self.mu[0] * (self._cov_cv_to_ca(self.sc_cv.P) + np.outer(dx0_comb, dx0_comb)) + \
                 self.mu[1] * (self.sc_ca.P + np.outer(dx1_comb, dx1_comb))
                 
        return x_comb, P_comb, self.mu, accepted


@dataclass
class LocalDetectionObservation:
    stamp_ns: int
    center_u: float
    center_v: float
    confidence: float
    class_id: int
    class_name: str

@dataclass(eq=False)
class SpatialObservation:
    stamp_ns: int
    origin: np.ndarray
    ray: np.ndarray


class Yolov8MainTriangulationNode(Node):
    def __init__(self) -> None:
        super().__init__('yolov8_main_triangulation_node')

        _weights_dir = FsPath(__file__).resolve().parent.parent / 'weights'
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
        _default_publish_markers = True
        _default_camera_frame_override = ''
        if _config_path.exists():
            import yaml
            try:
                with open(_config_path, 'r', encoding='utf-8') as _f:
                    _cfg = yaml.safe_load(_f)
                    if _cfg and 'vision' in _cfg:
                        _v = _cfg['vision']
                        _default_device = _v.get('device', _default_device)
                        _override_image_topic = _v.get('main_image_topic', '') or ''
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
                        _default_publish_markers = bool(_v.get('publish_rviz_markers', _default_publish_markers))
                        _default_camera_frame_override = _v.get('camera_frame_override', '') or ''
            except Exception:
                pass

        self.declare_parameter('uav_name', 'uav2')
        self.declare_parameter('observer1_name', 'uav1')
        self.declare_parameter('observer2_name', 'uav2')
        self.declare_parameter('target_name', 'uav3')
        self.declare_parameter('pose_topic', '/target/pose_estimate')
        self.declare_parameter('odometry_topic', '/target/odometry_estimate')
        self.declare_parameter('estimated_position_topic', '/target/position_estimate')
        self.declare_parameter('triangulation_raw_topic', '/target/triangulation_raw')
        self.declare_parameter('error_vector_topic', '/target/position_error')
        self.declare_parameter('target_frame_id', 'world')
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
        self.declare_parameter('max_pair_age_sec', 0.25)
        self.declare_parameter('triangulation_min_baseline_m', 0.25)
        self.declare_parameter('max_triangulation_error_m', 2.0)
        self.declare_parameter('max_target_gt_age_sec', 0.75)
        self.declare_parameter('processing_rate_hz', 100.0)
        self.declare_parameter('tf_timeout_sec', 0.05)
        self.declare_parameter('world_frame_suffix', 'world_origin')
        self.declare_parameter('kf_process_noise_acc', 1.5)
        self.declare_parameter('kf_process_noise_vel', 0.5)
        self.declare_parameter('imm_transition_prob', 0.95)
        self.declare_parameter('kf_measurement_noise', 0.05)
        self.declare_parameter('kf_adaptive_cov_max_multiplier', 50.0)
        self.declare_parameter('kf_initial_covariance', 10.0)
        self.declare_parameter('kf_mahalanobis_threshold', 9.0)
        self.declare_parameter('kf_prediction_horizon_sec', 1.0)
        self.declare_parameter('kf_prediction_steps', 25)
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
        self.declare_parameter('publish_rviz_markers', _default_publish_markers)
        self.declare_parameter('camera_frame_override', _default_camera_frame_override)

        self.uav_name = str(self.get_parameter('uav_name').value)
        self.observer1_name = str(self.get_parameter('observer1_name').value)
        self.observer2_name = str(self.get_parameter('observer2_name').value)
        self.target_name = str(self.get_parameter('target_name').value)

        self.image_topic = _override_image_topic or f'/{self.uav_name}/rgb/image_raw'
        self._use_compressed = self.image_topic.endswith('/compressed')
        _img_base = self.image_topic.replace('/compressed', '').rsplit('/image_raw', 1)[0]
        self.camera_info_topic = f'{_img_base}/camera_info'
        self.detection_topic = f'/{self.uav_name}/detection'
        self.target_gt_odom_topic = f'/{self.target_name}/hw_api/ground_truth'
        
        self.remote_origin_topic = f'/{self.observer2_name}/camera_origin'
        self.remote_ray_topic = f'/{self.observer2_name}/ray'

        self.pose_topic = str(self.get_parameter('pose_topic').value)
        self.odometry_topic = str(self.get_parameter('odometry_topic').value)
        self.estimated_position_topic = str(self.get_parameter('estimated_position_topic').value)
        self.triangulation_raw_topic = str(self.get_parameter('triangulation_raw_topic').value)
        self.error_vector_topic = str(self.get_parameter('error_vector_topic').value)
        self.target_frame_id = str(self.get_parameter('target_frame_id').value)
        self.model_path = os.path.expanduser(str(self.get_parameter('model_path').value))
        self.confidence_threshold = float(self.get_parameter('confidence_threshold').value)
        self.iou_threshold = float(self.get_parameter('iou_threshold').value)
        self.imgsz = int(self.get_parameter('imgsz').value)
        self.max_detections = int(self.get_parameter('max_detections').value)
        self.device = str(self.get_parameter('device').value)
        self.line_width = max(1, int(self.get_parameter('line_width').value))
        self.target_class_id = int(self.get_parameter('target_class_id').value)
        self.max_pair_age_ns = int(float(self.get_parameter('max_pair_age_sec').value) * 1_000_000_000.0)
        self.triangulation_min_baseline_m = float(self.get_parameter('triangulation_min_baseline_m').value)
        self.max_triangulation_error_m = float(self.get_parameter('max_triangulation_error_m').value)
        self.max_target_gt_age_ns = int(float(self.get_parameter('max_target_gt_age_sec').value) * 1_000_000_000.0)
        self.processing_rate_hz = max(0.5, float(self.get_parameter('processing_rate_hz').value))
        self.tf_timeout_sec = float(self.get_parameter('tf_timeout_sec').value)
        self.world_frame_suffix = str(self.get_parameter('world_frame_suffix').value)
        self.kf_process_noise_acc = float(self.get_parameter('kf_process_noise_acc').value)
        self.kf_process_noise_vel = float(self.get_parameter('kf_process_noise_vel').value)
        self.imm_transition_prob = float(self.get_parameter('imm_transition_prob').value)
        self.kf_measurement_noise = float(self.get_parameter('kf_measurement_noise').value)
        self.kf_adaptive_cov_max_multiplier = float(self.get_parameter('kf_adaptive_cov_max_multiplier').value)
        self.kf_initial_covariance = float(self.get_parameter('kf_initial_covariance').value)
        self.kf_mahalanobis_threshold = float(self.get_parameter('kf_mahalanobis_threshold').value)
        self.kf_prediction_horizon_sec = float(self.get_parameter('kf_prediction_horizon_sec').value)
        self.kf_prediction_steps = max(5, int(self.get_parameter('kf_prediction_steps').value))
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
        self.publish_rviz_markers = bool(self.get_parameter('publish_rviz_markers').value)
        self._camera_frame_override = str(self.get_parameter('camera_frame_override').value) or ''

        if self.target_frame_id == 'world':
            self.target_frame_id = f'{self.uav_name}/{self.world_frame_suffix}'
            
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
        self.target_gt_odom: Optional[Odometry] = None

        # Buffers for exact matching despite WiFi delays
        self.local_observations = collections.deque(maxlen=10)
        self.remote_observations = collections.deque(maxlen=10)

        # IMM Filter Initialization
        p_stay = self.imm_transition_prob
        p_switch = 1.0 - p_stay
        pi = np.array([
            [p_stay, p_switch],
            [p_switch * 0.5, 1.0 - (p_switch * 0.5)]
        ], dtype=np.float64)
        mu0 = np.array([0.5, 0.5], dtype=np.float64)
        self.imm = IMMFilter(pi, mu0)
        self.imm_initialized = False
        self.imm_last_stamp_ns: Optional[int] = None
        self.filtered_position = np.zeros(3)
        self.filtered_velocity = np.zeros(3)
        self.filtered_acceleration = np.zeros(3)
        self.combined_cov = np.eye(9)
        self.imm_probs = mu0.copy()

        self.last_error_log_time = time.time()
        self.frame_counter = 0
        self._dropped_frames = 0
        self.last_log_time = time.time()
        self._last_image_time = time.time()
        self._imm_cov_trace_reset_thresh = 500.0

        # ROI tracker state — main node uses IMM 3D→2D projection + 2D bbox fallback
        self._roi_last_bbox: Optional[Tuple[float, float, float, float]] = None
        self._roi_last_center: Optional[Tuple[float, float]] = None
        self._roi_last_stamp_ns: int = 0
        self._roi_frame_counter: int = 0
        self._roi_consecutive_misses: int = 0
        self._init_gate_buffer: collections.deque = collections.deque(maxlen=self.roi_init_n)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self._publish_camera_static_tf(_config_path)

        # Publishers
        self.detection_pub = self.create_publisher(CompressedImage, self.detection_topic + '/compressed', 10)
        self.pose_pub = self.create_publisher(PoseStamped, self.pose_topic, 10)
        self.odometry_pub = self.create_publisher(Odometry, self.odometry_topic, 10)
        self.estimated_position_pub = self.create_publisher(Vector3Stamped, self.estimated_position_topic, 10)
        self.triangulation_raw_pub = self.create_publisher(PointStamped, self.triangulation_raw_topic, 10)
        self.error_vector_pub = self.create_publisher(Vector3Stamped, self.error_vector_topic, 10)
        self.predicted_trajectory_pub = self.create_publisher(Path, '/target/predicted_trajectory', 10)
        self.trajectory_markers_pub = self.create_publisher(MarkerArray, '/target/predicted_trajectory_markers', 10)
        self.position_markers_pub = self.create_publisher(MarkerArray, '/target/position_markers', 10)

        # Subscribers — BEST_EFFORT for real-time sensor data over WiFi
        _sensor_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(Odometry, self.target_gt_odom_topic, self.target_gt_odom_callback, 20)
        if self._use_compressed:
            self.create_subscription(CompressedImage, self.image_topic, self.image_callback, _sensor_qos)
        else:
            self.create_subscription(Image, self.image_topic, self.image_callback, _sensor_qos)
        self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_callback, 10)
        
        # Exact Time Synchronizer for incoming remote topics
        self.origin_sub = message_filters.Subscriber(self, PointStamped, self.remote_origin_topic)
        self.ray_sub = message_filters.Subscriber(self, Vector3Stamped, self.remote_ray_topic)
        # Even though they are exact, a small slop helps if ROS 2 messaging tweaks the headers
        self.ts = message_filters.ApproximateTimeSynchronizer([self.origin_sub, self.ray_sub], 10, slop=0.01)
        self.ts.registerCallback(self.remote_sync_callback)

        self.create_timer(1.0 / self.processing_rate_hz, self.processing_timer_callback)

        self.get_logger().info(f'YOLO model path: {self.model_path}')
        self.get_logger().info(f'Main Triangulation Node initialization complete.')
        self.get_logger().info(f'Listening to local: {self.image_topic}')
        self.get_logger().info(f'Listening to remote: {self.remote_origin_topic}')

    def _publish_camera_static_tf(self, config_path: FsPath) -> None:
        """Publish static TF {uav_name}/fcu → rgbd_link from config."""
        try:
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f)
            ct = cfg.get('camera_tf', {})
            role_tf = ct.get('observer1', {})
            child = ct.get('child_frame', 'rgbd_link')
        except Exception:
            self.get_logger().warn('No camera_tf config found, skipping static TF')
            return

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = f'{self.uav_name}/fcu'
        t.child_frame_id = child
        t.transform.translation.x = float(role_tf.get('x', 0.0))
        t.transform.translation.y = float(role_tf.get('y', 0.0))
        t.transform.translation.z = float(role_tf.get('z', 0.0))
        r = role_tf.get('roll', 0.0)
        p = role_tf.get('pitch', 0.0)
        y = role_tf.get('yaw', 0.0)
        cy, sy = math.cos(y * 0.5), math.sin(y * 0.5)
        cp, sp = math.cos(p * 0.5), math.sin(p * 0.5)
        cr, sr = math.cos(r * 0.5), math.sin(r * 0.5)
        t.transform.rotation.w = cr * cp * cy + sr * sp * sy
        t.transform.rotation.x = sr * cp * cy - cr * sp * sy
        t.transform.rotation.y = cr * sp * cy + sr * cp * sy
        t.transform.rotation.z = cr * cp * sy - sr * sp * cy

        self._static_tf_broadcaster = StaticTransformBroadcaster(self)
        self._static_tf_broadcaster.sendTransform(t)
        self.get_logger().info(
            f'Static TF: {t.header.frame_id} -> {t.child_frame_id} '
            f'(xyz={t.transform.translation.x:.3f},{t.transform.translation.y:.3f},{t.transform.translation.z:.3f})'
        )

    def camera_info_callback(self, msg: CameraInfo) -> None:
        self.camera_info = msg

    def target_gt_odom_callback(self, msg: Odometry) -> None:
        self.target_gt_odom = msg

    def image_callback(self, msg) -> None:
        if self.latest_image is not None:
            self._dropped_frames += 1
        self.latest_image = msg
        self._last_image_time = time.time()

    def remote_sync_callback(self, origin_msg: PointStamped, ray_msg: Vector3Stamped) -> None:
        stamp_ns = stamp_to_ns(origin_msg.header.stamp)
        origin = np.array([origin_msg.point.x, origin_msg.point.y, origin_msg.point.z], dtype=np.float64)
        ray = np.array([ray_msg.vector.x, ray_msg.vector.y, ray_msg.vector.z], dtype=np.float64)
        
        self.remote_observations.append(SpatialObservation(stamp_ns, origin, ray))
        self.try_match_and_triangulate()

    def processing_timer_callback(self) -> None:
        msg = self.latest_image
        if msg is None or self.camera_info is None:
            if time.time() - self._last_image_time > 3.0:
                self.get_logger().warn('No image received for 3 s!', throttle_duration_sec=5.0)
            return

        stamp_ns = stamp_to_ns(msg.header.stamp)
        if stamp_ns <= self.last_processed_image_stamp_ns:
            # Coast: predict-only when IMM is initialized but no new measurement
            if self.imm_initialized:
                now_ns = self.get_clock().now().nanoseconds
                dt_since = (now_ns - self.imm_last_stamp_ns) / 1e9
                if dt_since > 0.2:
                    self._coast_predict(now_ns)
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
            self.get_logger().error(f'Failed to convert input image: {exc}')
            return

        stamp_ns = stamp_to_ns(msg.header.stamp)
        img_h, img_w = bgr_image.shape[:2]

        # Decide: ROI crop or full scan
        use_roi = False
        roi_x1 = roi_y1 = roi_x2 = roi_y2 = 0
        infer_image = bgr_image
        infer_imgsz = self.imgsz

        if self.roi_crop_enabled:
            force_full = (self._roi_frame_counter % self.roi_full_scan_interval == 0)

            roi_center = self._get_roi_center(stamp_ns, img_w, img_h)
            has_track = roi_center is not None

            if has_track and not force_full:
                pred_u, pred_v = roi_center

                # ROI size from last bbox or default — landscape (match image aspect ratio)
                if self._roi_last_bbox is not None:
                    bx1, by1, bx2, by2 = self._roi_last_bbox
                    roi_h = max(bx2 - bx1, by2 - by1) * self.roi_margin
                else:
                    roi_h = float(self.roi_min_size) * self.roi_margin
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

        # Update ROI tracker from detection
        if best_detection is not None:
            if self._roi_last_bbox is not None or self.imm_initialized:
                # Active track (or IMM guidance available) — update tracker, reset misses
                self._update_roi_tracker(best_detection, results, stamp_ns, roi_x1, roi_y1)
                self._roi_consecutive_misses = 0
            else:
                # No active track — apply M-of-N init gate
                if self._check_init_gate(best_detection.center_u, best_detection.center_v, stamp_ns):
                    self._update_roi_tracker(best_detection, results, stamp_ns, roi_x1, roi_y1)
                    self._roi_consecutive_misses = 0
                    self._init_gate_buffer.clear()
                    self.get_logger().info('ROI track initialized via M-of-N gate')
        elif self._roi_last_bbox is not None:
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
            spatial_obs = self.compute_ray(best_detection)
            if spatial_obs is not None:
                self.local_observations.append(spatial_obs)
                self.try_match_and_triangulate()

        self.frame_counter += 1
        now = time.time()
        if now - self.last_log_time >= 2.0:
            fps = self.frame_counter / (now - self.last_log_time)
            mode = 'ROI' if use_roi else 'FULL'
            esc = f' esc={self._roi_consecutive_misses}' if self._roi_consecutive_misses > 0 else ''
            self.get_logger().info(f'Main node: FPS={fps:.2f}, inference={infer_ms:.1f}ms, mode={mode}{esc}, dropped={self._dropped_frames}')
            self.frame_counter = 0
            self._dropped_frames = 0
            self.last_log_time = now

    def _get_roi_center(self, stamp_ns: int, img_w: int, img_h: int) -> Optional[Tuple[float, float]]:
        """Get ROI center by projecting IMM 3D prediction to 2D, falling back to 2D bbox tracker."""
        # Strategy 1: Use IMM 3D position → project to image plane
        if self.imm_initialized and self.camera_info is not None:
            camera_frame = self._camera_frame_override or self.camera_info.header.frame_id
            try:
                transform = self.tf_buffer.lookup_transform(
                    camera_frame, self.target_frame_id, Time(), Duration(seconds=0)
                )
                t = transform.transform.translation
                q = transform.transform.rotation
                r_cw = quat_to_rotmat(q.x, q.y, q.z, q.w)
                t_cw = np.array([t.x, t.y, t.z], dtype=np.float64)

                # IMM predicted 3D position with velocity extrapolation
                dt = (stamp_ns - self.imm_last_stamp_ns) / 1_000_000_000.0
                pred_3d = self.filtered_position + self.filtered_velocity * dt

                # Transform world→camera
                p_cam = r_cw @ pred_3d + t_cw
                if p_cam[2] > 0.1:  # target must be in front of camera
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

        # Strategy 2: Fallback to 2D bbox tracker
        if self._roi_last_center is not None and (stamp_ns - self._roi_last_stamp_ns) < self.roi_timeout_ns:
            return self._roi_last_center

        return None

    def _check_init_gate(self, center_u: float, center_v: float, stamp_ns: int) -> bool:
        """M-of-N init gate: require M consistent detections out of N before starting track."""
        self._init_gate_buffer.append((center_u, center_v, stamp_ns))
        if len(self._init_gate_buffer) < self.roi_init_m:
            return False
        centers = [(u, v) for u, v, _ in self._init_gate_buffer]
        for i in range(len(centers)):
            ref_u, ref_v = centers[i]
            close = sum(1 for u, v in centers if abs(u - ref_u) < self.roi_init_spread and abs(v - ref_v) < self.roi_init_spread)
            if close >= self.roi_init_m:
                return True
        return False

    def _update_roi_tracker(self, det: 'LocalDetectionObservation', results, stamp_ns: int, roi_x1: int, roi_y1: int) -> None:
        """Update 2D ROI tracker from detection bbox."""
        result = results[0]
        boxes_xyxy = result.boxes.xyxy.tolist()
        confs = result.boxes.conf.tolist()
        best_idx = int(np.argmax(confs))
        bx1, by1, bx2, by2 = boxes_xyxy[best_idx]
        bx1 += roi_x1; by1 += roi_y1
        bx2 += roi_x1; by2 += roi_y1

        self._roi_last_bbox = (bx1, by1, bx2, by2)
        self._roi_last_center = (det.center_u, det.center_v)
        self._roi_last_stamp_ns = stamp_ns

    def extract_best_detection(self, results, msg: Image, roi_pred_center: Optional[Tuple[float, float]] = None,
                               roi_x1: int = 0, roi_y1: int = 0, roi_half_side: float = 1.0) -> Optional[LocalDetectionObservation]:
        if not results:
            return None

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return None

        names = result.names if result.names is not None else {}
        boxes_xyxy: List[Tuple[float, float, float, float]] = result.boxes.xyxy.tolist()
        confidences: List[float] = result.boxes.conf.tolist()
        classes: List[float] = result.boxes.cls.tolist()

        candidates: List[LocalDetectionObservation] = []
        for box, conf, cls_idx in zip(boxes_xyxy, confidences, classes):
            class_id = int(cls_idx)
            if self.target_class_id >= 0 and class_id != self.target_class_id:
                continue

            x1, y1, x2, y2 = box
            center_u = 0.5 * (x1 + x2)
            center_v = 0.5 * (y1 + y2)
            class_name = names.get(class_id, str(class_id)) if isinstance(names, dict) else str(class_id)
            candidates.append(
                LocalDetectionObservation(
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
                d_pixel = ((det.center_u + roi_x1 - pred_u) ** 2 + (det.center_v + roi_y1 - pred_v) ** 2) ** 0.5
                return det.confidence - self.roi_sel_lambda * min(d_pixel / d_max, 1.0)
            return max(candidates, key=score)

        return max(candidates, key=lambda item: item.confidence)

    def draw_predictions(self, image: np.ndarray, results, best_detection: Optional[LocalDetectionObservation], use_roi: bool = False, roi_x1: int = 0, roi_y1: int = 0, roi_x2: int = 0, roi_y2: int = 0) -> np.ndarray:
        annotated = image.copy()
        if not results:
            return annotated
        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            if use_roi and roi_x2 > roi_x1 and roi_y2 > roi_y1:
                cv2.rectangle(annotated, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 220, 0), 2)
                cv2.putText(annotated, 'ROI', (roi_x1, roi_y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1, cv2.LINE_AA)
            return annotated
        names = result.names if result.names is not None else {}
        boxes_xyxy: List[Tuple[float, float, float, float]] = result.boxes.xyxy.tolist()
        confidences: List[float] = result.boxes.conf.tolist()
        classes: List[float] = result.boxes.cls.tolist()

        for box, conf, cls_idx in zip(boxes_xyxy, confidences, classes):
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

    def compute_ray(self, obs: LocalDetectionObservation) -> Optional[SpatialObservation]:
        fx = float(self.camera_info.k[0])
        fy = float(self.camera_info.k[4])
        cx = float(self.camera_info.k[2])
        cy = float(self.camera_info.k[5])

        ray_optical = normalize(np.array([(obs.center_u - cx) / fx, (obs.center_v - cy) / fy, 1.0], dtype=np.float64))
        camera_frame = self._camera_frame_override or self.camera_info.header.frame_id

        try:
            sec = obs.stamp_ns // 1_000_000_000
            nanosec = obs.stamp_ns % 1_000_000_000
            lookup_time = Time(seconds=sec, nanoseconds=nanosec)
            transform = self.tf_buffer.lookup_transform(
                self.target_frame_id, camera_frame, lookup_time, Duration(seconds=0),
            )
        except Exception:
            # Exact timestamp not available yet — use latest TF (small pose error, never blocks)
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.target_frame_id, camera_frame, Time(), Duration(seconds=0),
                )
            except Exception as exc:
                self.get_logger().warn(f'TF lookup failed: {exc}', throttle_duration_sec=2.0)
                return None

        t = transform.transform.translation
        q = transform.transform.rotation
        camera_origin_world = np.array([t.x, t.y, t.z], dtype=np.float64)
        r_wc = quat_to_rotmat(q.x, q.y, q.z, q.w)
        ray_world = normalize(r_wc @ ray_optical)

        return SpatialObservation(stamp_ns=obs.stamp_ns, origin=camera_origin_world, ray=ray_world)

    def try_match_and_triangulate(self) -> None:
        if not self.local_observations or not self.remote_observations:
            return

        best_loc = None
        best_rem = None
        best_diff = self.max_pair_age_ns
        
        for loc in self.local_observations:
            for rem in self.remote_observations:
                diff = abs(loc.stamp_ns - rem.stamp_ns)
                if diff <= best_diff:
                    best_diff = diff
                    best_loc = loc
                    best_rem = rem
                
        if best_loc is not None and best_rem is not None and best_diff <= self.max_pair_age_ns:
            self.local_observations.remove(best_loc)
            self.remote_observations.remove(best_rem)
            
            # Use Kinematic Ray Shifting to compensate for asynchronous hardware timers
            if self.imm_initialized:
                dt_sec = (best_loc.stamp_ns - best_rem.stamp_ns) / 1_000_000_000.0
                # Shift the remote drone's origin forward/backward by the target's estimated 
                # motion during the time gap, projecting the remote ray to the local timestamp!
                adjusted_rem_origin = best_rem.origin + self.filtered_velocity * dt_sec
                self.execute_triangulation(best_loc.origin, best_loc.ray, adjusted_rem_origin, best_rem.ray, best_loc.stamp_ns)
            else:
                self.execute_triangulation(best_loc.origin, best_loc.ray, best_rem.origin, best_rem.ray, max(best_loc.stamp_ns, best_rem.stamp_ns))

    def execute_triangulation(self, origin1: np.ndarray, dir1: np.ndarray, origin2: np.ndarray, dir2: np.ndarray, stamp_ns: int) -> None:
        baseline = float(np.linalg.norm(origin2 - origin1))
        if baseline < self.triangulation_min_baseline_m:
            return

        midpoint, error = triangulate_midpoint(origin1, dir1, origin2, dir2)
        if error > self.max_triangulation_error_m:
            self.get_logger().warn(
                f'Triangulation rejected: ray mismatch {error:.2f} m exceeds {self.max_triangulation_error_m:.2f} m'
            )
            return

        measurement = midpoint

        raw_msg = PointStamped()
        raw_msg.header.frame_id = self.target_frame_id
        raw_msg.header.stamp.sec = stamp_ns // 1_000_000_000
        raw_msg.header.stamp.nanosec = stamp_ns % 1_000_000_000
        raw_msg.point.x = float(midpoint[0])
        raw_msg.point.y = float(midpoint[1])
        raw_msg.point.z = float(midpoint[2])
        self.triangulation_raw_pub.publish(raw_msg)

        cos_angle = np.clip(np.dot(dir1, dir2), -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)
        sin_angle = np.clip(np.sin(angle_rad), 1e-6, 1.0)
        
        bisector = normalize(dir1 + dir2)
        var_base = self.kf_measurement_noise
        multiplier = min(1.0 / sin_angle, self.kf_adaptive_cov_max_multiplier)
        var_depth = var_base * multiplier
        
        v_diff = normalize(dir1 - dir2)
        v_cross = normalize(np.cross(bisector, v_diff))
        
        P_eig = np.array([
            [var_depth, 0.0, 0.0],
            [0.0, var_base, 0.0],
            [0.0, 0.0, var_base]
        ])
        Evecs = np.column_stack((bisector, v_diff, v_cross))
        R = Evecs @ P_eig @ Evecs.T

        if not self.imm_initialized:
            self.imm.initialize(measurement, self.kf_initial_covariance)
            self.imm_initialized = True
            self.imm_last_stamp_ns = stamp_ns
            self.combined_cov = np.eye(9) * self.kf_initial_covariance
            self.filtered_position = measurement.copy()
        else:
            dt = (stamp_ns - self.imm_last_stamp_ns) / 1_000_000_000.0
            if dt > 0.0:
                Q_cv = build_process_noise_cv(dt, self.kf_process_noise_vel)
                Q_ca = build_process_noise_ca(dt, self.kf_process_noise_acc)
                x_comb, self.combined_cov, self.imm_probs, accepted = self.imm.predict_and_correct(dt, Q_cv, Q_ca, measurement, R, mahalanobis_thresh=self.kf_mahalanobis_threshold)
                
                if not accepted:
                    self.get_logger().warn("Outlier measurement rejected by Mahalanobis gating! Filter coasting purely on kinetics.")
                
                self.filtered_position = x_comb[:3].copy()
                self.filtered_velocity = x_comb[3:6].copy()
                self.filtered_acceleration = x_comb[6:9].copy()
                self.imm_last_stamp_ns = stamp_ns

                # Fix #10: reset IMM if covariance diverges
                cov_trace = float(np.trace(self.combined_cov))
                if cov_trace > self._imm_cov_trace_reset_thresh:
                    self.get_logger().warn(f'IMM covariance diverged (trace={cov_trace:.1f}), resetting filter at current measurement.')
                    self.imm.initialize(measurement, self.kf_initial_covariance)
                    self.imm_initialized = True
                    self.combined_cov = np.eye(9) * self.kf_initial_covariance
                    self.filtered_position = measurement.copy()
                    self.filtered_velocity = np.zeros(3)
                    self.filtered_acceleration = np.zeros(3)

        self.publish_pose_estimate(self.filtered_position, self.filtered_velocity, self.filtered_acceleration, stamp_ns)
        self.publish_ground_truth_error(self.filtered_position, stamp_ns)

    def _coast_predict(self, now_ns: int) -> None:
        """Predict-only step when no new triangulation is available (WiFi dropout)."""
        dt = (now_ns - self.imm_last_stamp_ns) / 1e9
        if dt <= 0.0 or dt > 5.0:
            return
        # Simple kinematic extrapolation from last filtered state
        self.filtered_position = self.filtered_position + self.filtered_velocity * dt + 0.5 * self.filtered_acceleration * dt * dt
        self.filtered_velocity = self.filtered_velocity + self.filtered_acceleration * dt
        self.imm_last_stamp_ns = now_ns
        self.get_logger().info(f'Coast predict: dt={dt:.2f}s, no measurement', throttle_duration_sec=2.0)
        self.publish_pose_estimate(self.filtered_position, self.filtered_velocity, self.filtered_acceleration, now_ns)

    def publish_pose_estimate(self, position: np.ndarray, velocity: np.ndarray, acceleration: np.ndarray, stamp_ns: int) -> None:
        sec = stamp_ns // 1_000_000_000
        nanosec = stamp_ns % 1_000_000_000

        estimated_position_msg = Vector3Stamped()
        estimated_position_msg.header.frame_id = self.target_frame_id
        estimated_position_msg.header.stamp.sec = int(sec)
        estimated_position_msg.header.stamp.nanosec = int(nanosec)
        estimated_position_msg.vector.x = float(position[0])
        estimated_position_msg.vector.y = float(position[1])
        estimated_position_msg.vector.z = float(position[2])
        self.estimated_position_pub.publish(estimated_position_msg)

        pose_msg = PoseStamped()
        pose_msg.header.frame_id = self.target_frame_id
        pose_msg.header.stamp.sec = int(sec)
        pose_msg.header.stamp.nanosec = int(nanosec)
        pose_msg.pose.position.x = float(position[0])
        pose_msg.pose.position.y = float(position[1])
        pose_msg.pose.position.z = float(position[2])
        pose_msg.pose.orientation.w = 1.0
        self.pose_pub.publish(pose_msg)

        P = self.combined_cov
        odom_msg = Odometry()
        odom_msg.header = pose_msg.header
        odom_msg.child_frame_id = 'target_estimate'
        odom_msg.pose.pose = pose_msg.pose
        odom_msg.pose.covariance[0] = float(P[0, 0])
        odom_msg.pose.covariance[7] = float(P[1, 1])
        odom_msg.pose.covariance[14] = float(P[2, 2])
        odom_msg.twist.twist.linear.x = float(velocity[0])
        odom_msg.twist.twist.linear.y = float(velocity[1])
        odom_msg.twist.twist.linear.z = float(velocity[2])
        odom_msg.twist.covariance[0] = float(P[3, 3])
        odom_msg.twist.covariance[7] = float(P[4, 4])
        odom_msg.twist.covariance[14] = float(P[5, 5])
        self.odometry_pub.publish(odom_msg)

        self._publish_predicted_trajectory(position, velocity, acceleration, stamp_ns)
        if self.publish_rviz_markers:
            self._publish_position_markers(position, stamp_ns)

    def _publish_predicted_trajectory(self, position: np.ndarray, velocity: np.ndarray, acceleration: np.ndarray, stamp_ns: int) -> None:
        sec = int(stamp_ns // 1_000_000_000)
        nanosec = int(stamp_ns % 1_000_000_000)

        path_msg = Path()
        path_msg.header.frame_id = self.target_frame_id
        path_msg.header.stamp.sec = sec
        path_msg.header.stamp.nanosec = nanosec

        dt_step = self.kf_prediction_horizon_sec / self.kf_prediction_steps
        pred_points: List[np.ndarray] = []
        
        if hasattr(self, 'imm') and self.imm_initialized:
            cur_mu = self.imm.mu.copy()
            x_cv = self.imm.sc_cv.x.copy()
            x_ca = self.imm.sc_ca.x.copy()
            
            A_cv = build_transition_matrix_cv(dt_step)
            A_ca = build_transition_matrix_ca(dt_step)
            
            for i in range(self.kf_prediction_steps + 1):
                pos_cv = x_cv[:3]
                pos_ca = x_ca[:3]
                pred_pos = pos_cv * cur_mu[0] + pos_ca * cur_mu[1]
                pred_points.append(pred_pos)
                
                x_cv = A_cv @ x_cv
                x_ca = A_ca @ x_ca
        else:
            for i in range(self.kf_prediction_steps + 1):
                t = i * dt_step
                pred_pos = position + velocity * t + 0.5 * acceleration * (t * t)
                pred_points.append(pred_pos)

        for i, pred_pos in enumerate(pred_points):
            t = i * dt_step
            pred_stamp_ns = stamp_ns + int(t * 1_000_000_000)
            pose = PoseStamped()
            pose.header.frame_id = self.target_frame_id
            pose.header.stamp.sec = int(pred_stamp_ns // 1_000_000_000)
            pose.header.stamp.nanosec = int(pred_stamp_ns % 1_000_000_000)
            pose.pose.position.x = float(pred_pos[0])
            pose.pose.position.y = float(pred_pos[1])
            pose.pose.position.z = float(pred_pos[2])
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)

        self.predicted_trajectory_pub.publish(path_msg)

        if not self.publish_rviz_markers:
            return

        marker_array = MarkerArray()
        clear = Marker()
        clear.header.frame_id = self.target_frame_id
        clear.header.stamp.sec = sec
        clear.header.stamp.nanosec = nanosec
        clear.ns = 'predicted_traj'
        clear.action = Marker.DELETEALL
        marker_array.markers.append(clear)

        n_segments = len(pred_points) - 1
        for i in range(n_segments):
            alpha = i / max(1, n_segments - 1)
            m = Marker()
            m.header.frame_id = self.target_frame_id
            m.header.stamp.sec = sec
            m.header.stamp.nanosec = nanosec
            m.ns = 'predicted_traj'
            m.id = i + 1
            m.type = Marker.ARROW
            m.action = Marker.ADD
            m.scale.x = 0.06
            m.scale.y = 0.14
            m.scale.z = 0.0
            m.color.r = 1.0
            m.color.g = float(1.0 - alpha)
            m.color.b = 0.0
            m.color.a = float(0.9 - 0.45 * alpha)
            m.lifetime.sec = 1
            p0 = pred_points[i]
            p1 = pred_points[i + 1]
            m.points = [
                Point(x=float(p0[0]), y=float(p0[1]), z=float(p0[2])),
                Point(x=float(p1[0]), y=float(p1[1]), z=float(p1[2])),
            ]
            marker_array.markers.append(m)

        self.trajectory_markers_pub.publish(marker_array)

    def _publish_position_markers(self, position: np.ndarray, stamp_ns: int) -> None:
        sec = int(stamp_ns // 1_000_000_000)
        nanosec = int(stamp_ns % 1_000_000_000)
        marker_array = MarkerArray()

        est = Marker()
        est.header.frame_id = self.target_frame_id
        est.header.stamp.sec = sec
        est.header.stamp.nanosec = nanosec
        est.ns = 'position_estimate'
        est.id = 0
        est.type = Marker.SPHERE
        est.action = Marker.ADD
        est.pose.position.x = float(position[0])
        est.pose.position.y = float(position[1])
        est.pose.position.z = float(position[2])
        est.pose.orientation.w = 1.0
        est.scale.x = est.scale.y = est.scale.z = 0.4
        est.color.r = 0.0
        est.color.g = 0.4
        est.color.b = 1.0
        est.color.a = 0.9
        est.lifetime.sec = 1
        marker_array.markers.append(est)

        if self.target_gt_odom is not None:
            gt_stamp_ns = stamp_to_ns(self.target_gt_odom.header.stamp)
            if abs(stamp_ns - gt_stamp_ns) <= self.max_target_gt_age_ns:
                gt = Marker()
                gt.header.frame_id = self.target_frame_id
                gt.header.stamp.sec = sec
                gt.header.stamp.nanosec = nanosec
                gt.ns = 'position_ground_truth'
                gt.id = 1
                gt.type = Marker.SPHERE
                gt.action = Marker.ADD
                gt.pose.position.x = float(self.target_gt_odom.pose.pose.position.x)
                gt.pose.position.y = float(self.target_gt_odom.pose.pose.position.y)
                gt.pose.position.z = float(self.target_gt_odom.pose.pose.position.z)
                gt.pose.orientation.w = 1.0
                gt.scale.x = gt.scale.y = gt.scale.z = 0.4
                gt.color.r = 0.0
                gt.color.g = 1.0
                gt.color.b = 0.0
                gt.color.a = 0.9
                gt.lifetime.sec = 1
                marker_array.markers.append(gt)

        self.position_markers_pub.publish(marker_array)

    def publish_ground_truth_error(self, estimated_position: np.ndarray, stamp_ns: int) -> None:
        if self.target_gt_odom is None:
            return

        gt_stamp_ns = stamp_to_ns(self.target_gt_odom.header.stamp)
        if abs(stamp_ns - gt_stamp_ns) > self.max_target_gt_age_ns:
            return

        gt_position = np.array(
            [
                self.target_gt_odom.pose.pose.position.x,
                self.target_gt_odom.pose.pose.position.y,
                self.target_gt_odom.pose.pose.position.z,
            ],
            dtype=np.float64,
        )
        error = estimated_position - gt_position
        sec = stamp_ns // 1_000_000_000
        nanosec = stamp_ns % 1_000_000_000

        error_vector_msg = Vector3Stamped()
        error_vector_msg.header.frame_id = self.target_frame_id
        error_vector_msg.header.stamp.sec = int(sec)
        error_vector_msg.header.stamp.nanosec = int(nanosec)
        error_vector_msg.vector.x = float(error[0])
        error_vector_msg.vector.y = float(error[1])
        error_vector_msg.vector.z = float(error[2])
        self.error_vector_pub.publish(error_vector_msg)

        now = time.time()
        if now - self.last_error_log_time >= 1.0:
            self.get_logger().info(
                'Estimated pos: x=%.3f, y=%.3f, z=%.3f | GT error: dx=%.3f, dy=%.3f, dz=%.3f'
                % (estimated_position[0], estimated_position[1], estimated_position[2], error[0], error[1], error[2])
            )
            self.last_error_log_time = now

def main() -> None:
    rclpy.init()
    node = Yolov8MainTriangulationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
