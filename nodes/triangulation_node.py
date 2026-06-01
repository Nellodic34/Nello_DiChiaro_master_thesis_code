#!/usr/bin/env python3

import collections
import struct
import time
from dataclasses import dataclass
from pathlib import Path as FsPath
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, PointStamped, PoseStamped, Vector3Stamped
from nav_msgs.msg import Odometry, Path
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from std_msgs.msg import UInt8MultiArray
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray


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


def project_ray_on_image(image, camera_info, transform_wc, observations, color_list=None, thickness=2, length=10.0):
    """Project matched world-frame rays back onto observer1's image for triangulation debug."""
    img = image.copy()
    camera_matrix = np.array(camera_info.k).reshape(3, 3)

    translation = transform_wc.transform.translation
    rotation = transform_wc.transform.rotation

    rotation_cw = quat_to_rotmat(rotation.x, rotation.y, rotation.z, rotation.w)
    translation_cw = np.array([translation.x, translation.y, translation.z], dtype=np.float64)

    for index, (origin, ray) in enumerate(observations):
        color = (0, 0, 255) if color_list is None else color_list[index % len(color_list)]

        ray_dir = ray / np.linalg.norm(ray)
        point_start_world = origin + ray_dir * 1.0
        point_end_world = origin + ray_dir * length

        point_start_camera = rotation_cw @ point_start_world + translation_cw
        point_end_camera = rotation_cw @ point_end_world + translation_cw

        if point_start_camera[2] <= 0 or point_end_camera[2] <= 0:
            continue

        points_camera = np.stack([point_start_camera, point_end_camera], axis=0).T
        points_2d = camera_matrix @ points_camera
        points_2d = points_2d[:2] / points_2d[2]

        pt1 = tuple(np.round(points_2d[:, 0]).astype(int))
        pt2 = tuple(np.round(points_2d[:, 1]).astype(int))

        cv2.line(img, pt1, pt2, color, thickness, lineType=cv2.LINE_AA)
        cv2.circle(img, pt1, 5, color, -1)

    return img


def stamp_to_ns(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-9:
        raise ValueError('Cannot normalize near-zero vector')
    return vec / norm


 # Packet layout: [stamp_ns, observation_id, confidence, origin xyz, ray xyz, tf_delta_ns, tf_status].
OBSERVATION_PACKET_STRUCT = struct.Struct('<qIf3f3fqB')
TF_STATUS_EXACT = 0


def unpack_observation_packet(msg: UInt8MultiArray) -> Optional[Tuple[int, int, float, np.ndarray, np.ndarray, int, int]]:
    """Unpack a compact observer measurement packet from the network topic."""
    packet_bytes = bytes(msg.data)
    if len(packet_bytes) != OBSERVATION_PACKET_STRUCT.size:
        return None

    unpacked = OBSERVATION_PACKET_STRUCT.unpack(packet_bytes)
    stamp_ns, observation_id, confidence = unpacked[:3]
    origin = np.array(unpacked[3:6], dtype=np.float64)
    ray = np.array(unpacked[6:9], dtype=np.float64)
    tf_delta_ns = int(unpacked[9])
    tf_status = int(unpacked[10])
    return int(stamp_ns), int(observation_id), float(confidence), origin, ray, tf_delta_ns, tf_status


def triangulate_midpoint(
    origin_a: np.ndarray,
    dir_a: np.ndarray,
    origin_b: np.ndarray,
    dir_b: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Return the midpoint of the shortest segment between two 3D rays and its residual gap."""
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

    def predict(self, sc: StateCov, u: np.ndarray, Q: np.ndarray) -> StateCov:
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
    A[0, 3] = dt
    A[0, 6] = dt2
    A[1, 4] = dt
    A[1, 7] = dt2
    A[2, 5] = dt
    A[2, 8] = dt2
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

    def predict_and_correct(
        self,
        dt: float,
        Q_cv: np.ndarray,
        Q_ca: np.ndarray,
        z: np.ndarray,
        R: np.ndarray,
        mahalanobis_thresh: float = 9.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
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
        P0_mix_cv = (
            mu_mix[0, 0] * (P_cv + np.outer(dx0, dx0))
            + mu_mix[1, 0] * (self._cov_ca_to_cv(P_ca) + np.outer(dx1, dx1))
        )

        dx0_ca = x_cv_as_ca - x0_mix_ca
        dx1_ca = x_ca - x0_mix_ca
        P0_mix_ca = (
            mu_mix[0, 1] * (self._cov_cv_to_ca(P_cv) + np.outer(dx0_ca, dx0_ca))
            + mu_mix[1, 1] * (P_ca + np.outer(dx1_ca, dx1_ca))
        )

        self.sc_cv.x = x0_mix_cv
        self.sc_cv.P = P0_mix_cv
        self.sc_ca.x = x0_mix_ca
        self.sc_ca.P = P0_mix_ca

        u_cv = np.zeros(1, dtype=np.float64)
        u_ca = np.zeros(1, dtype=np.float64)

        self.kf_cv.A = build_transition_matrix_cv(dt)
        sc_pred_cv = self.kf_cv.predict(self.sc_cv, u_cv, Q_cv)

        self.kf_ca.A = build_transition_matrix_ca(dt)
        sc_pred_ca = self.kf_ca.predict(self.sc_ca, u_ca, Q_ca)

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
            P_comb = (
                self.mu[0] * (self._cov_cv_to_ca(self.sc_cv.P) + np.outer(dx0_comb, dx0_comb))
                + self.mu[1] * (self.sc_ca.P + np.outer(dx1_comb, dx1_comb))
            )
            return x_comb, P_comb, self.mu, accepted

        def mv_normal_pdf_fixed(y, S_fixed):
            try:
                invS = np.linalg.inv(S_fixed)
                detS = max(np.linalg.det(S_fixed), 1e-12)
            except np.linalg.LinAlgError:
                return 1e-12
            exponent = -0.5 * y.T @ invS @ y
            # Normalization constant is required in IMM to prevent structural bias
            # towards models with higher covariance (like CA).
            k = y.shape[0]
            norm_const = 1.0 / np.sqrt(((2 * np.pi) ** k) * detS)
            return float(norm_const * np.exp(max(-50.0, min(50.0, exponent))))

        # Per Blom & Bar-Shalom (1988), the IMM likelihood must use the
        # innovation covariance S_j = H P_pred H^T + R, not just R.
        # Using R alone ignores the filter's own prediction uncertainty.
        Lambda = np.array([mv_normal_pdf_fixed(y_cv, S_cv), mv_normal_pdf_fixed(y_ca, S_ca)])

        self.sc_cv = self.kf_cv.correct(sc_pred_cv, z, R)
        self.sc_ca = self.kf_ca.correct(sc_pred_ca, z, R)

        self.mu = Lambda * c_bar
        self.mu /= np.sum(self.mu) + 1e-12

        x_comb = self._state_cv_to_ca(self.sc_cv.x) * self.mu[0] + self.sc_ca.x * self.mu[1]
        dx0_comb = self._state_cv_to_ca(self.sc_cv.x) - x_comb
        dx1_comb = self.sc_ca.x - x_comb
        P_comb = (
            self.mu[0] * (self._cov_cv_to_ca(self.sc_cv.P) + np.outer(dx0_comb, dx0_comb))
            + self.mu[1] * (self.sc_ca.P + np.outer(dx1_comb, dx1_comb))
        )

        return x_comb, P_comb, self.mu, accepted


@dataclass(eq=False)
class SpatialObservation:
    """Frozen observer measurement carried through matching, frame transforms, and debug."""
    stamp_ns: int
    origin: np.ndarray
    ray: np.ndarray
    frame_id: str
    source_name: str
    observation_id: int = 0
    confidence: float = 0.0
    tf_delta_ns: int = 0
    tf_status: int = TF_STATUS_EXACT


@dataclass(eq=False)
class DebugImageFrame:
    stamp_ns: int
    bgr_image: np.ndarray
    header: object


@dataclass(eq=False)
class MatchedObservationPair:
    local_obs: SpatialObservation
    remote_obs: SpatialObservation
    aligned_local_obs: SpatialObservation
    aligned_remote_obs: SpatialObservation
    aligned_stamp_ns: int
    pair_diff_ns: int
    accepted: bool


class TriangulationNode(Node):
    def __init__(self) -> None:
        super().__init__('triangulation_node')

        _config_path = FsPath(__file__).resolve().parent.parent / 'config' / 'runtime_stack.yaml'
        _observer1_name = 'uav1'
        _observer2_name = 'uav2'
        _target_name = 'uav3'
        _world_frame_suffix = 'world_origin'
        _target_gt_odom_suffix = 'hw_api/ground_truth'
        _default_publish_markers = True
        _default_publish_debug_annotated = False
        _observer1_image_topic = ''
        _observer2_image_topic = ''
        _observer1_camera_info_topic = ''
        _default_observation_topic_name = 'target_observation'
        _default_max_tf_delta_ms = 30.0
        _default_process_noise_acc = 1.5
        _default_process_noise_vel = 0.5
        _default_imm_transition_prob = 0.95
        _default_measurement_noise = 0.05
        _default_adaptive_cov_max_multiplier = 50.0
        _default_initial_covariance = 10.0
        _default_mahalanobis_threshold = 9.0
        if _config_path.exists():
            import yaml
            try:
                with open(_config_path, 'r', encoding='utf-8') as _f:
                    _cfg = yaml.safe_load(_f)
                    if _cfg:
                        if 'uav_roles' in _cfg:
                            _roles = _cfg['uav_roles']
                            _observer1_name = _roles.get('observer1', _observer1_name)
                            _observer2_name = _roles.get('observer2', _observer2_name)
                            _target_name = _roles.get('target', _target_name)
                        if 'evaluation' in _cfg:
                            _eval = _cfg['evaluation']
                            _world_frame_suffix = _eval.get('world_frame_suffix', _world_frame_suffix)
                            _target_gt_odom_suffix = _eval.get('odom_topic_suffix', _target_gt_odom_suffix)
                        if 'vision' in _cfg:
                            _vision = _cfg['vision']
                            _default_publish_markers = bool(
                                _vision.get('publish_rviz_markers', _default_publish_markers)
                            )
                            _default_publish_debug_annotated = bool(
                                _vision.get('publish_debug_annotated_image', _default_publish_debug_annotated)
                            )
                            _observer1_image_topic = _vision.get('main_image_topic', '') or ''
                            _observer2_image_topic = _vision.get('observer_image_topic', '') or ''
                            _observer1_camera_info_topic = _vision.get('main_camera_info_topic', '') or ''
                            _default_observation_topic_name = (
                                _vision.get('observation_topic_name', _default_observation_topic_name)
                                or _default_observation_topic_name
                            )
                            _default_max_tf_delta_ms = float(
                                _vision.get('max_tf_delta_ms', _default_max_tf_delta_ms)
                            )
                        if 'ekf' in _cfg:
                            _ekf = _cfg['ekf']
                            _default_process_noise_acc = float(
                                _ekf.get('kf_process_noise_acc', _default_process_noise_acc)
                            )
                            _default_process_noise_vel = float(
                                _ekf.get('kf_process_noise_vel', _default_process_noise_vel)
                            )
                            _default_imm_transition_prob = float(
                                _ekf.get('imm_transition_prob', _default_imm_transition_prob)
                            )
                            _default_measurement_noise = float(
                                _ekf.get('kf_measurement_noise', _default_measurement_noise)
                            )
                            _default_adaptive_cov_max_multiplier = float(
                                _ekf.get(
                                    'kf_adaptive_cov_max_multiplier',
                                    _default_adaptive_cov_max_multiplier,
                                )
                            )
                            _default_initial_covariance = float(
                                _ekf.get('kf_initial_covariance', _default_initial_covariance)
                            )
                            _default_mahalanobis_threshold = float(
                                _ekf.get('kf_mahalanobis_threshold', _default_mahalanobis_threshold)
                            )
            except Exception:
                pass

        self.declare_parameter('observer1_name', _observer1_name)
        self.declare_parameter('observer2_name', _observer2_name)
        self.declare_parameter('target_name', _target_name)
        self.declare_parameter('pose_topic', '/target/pose_estimate')
        self.declare_parameter('odometry_topic', '/target/odometry_estimate')
        self.declare_parameter('estimated_position_topic', '/target/position_estimate')
        self.declare_parameter('triangulation_raw_topic', '/target/triangulation_raw')
        self.declare_parameter('error_vector_topic', '/target/position_error')
        self.declare_parameter('target_frame_id', 'world')
        self.declare_parameter('max_pair_age_sec', 0.25)
        self.declare_parameter('observation_buffer_sec', 1.0)
        self.declare_parameter('triangulation_min_baseline_m', 0.25)
        self.declare_parameter('max_triangulation_error_m', 2.0)
        self.declare_parameter('max_target_gt_age_sec', 0.75)
        self.declare_parameter('processing_rate_hz', 100.0)
        self.declare_parameter('tf_timeout_sec', 0.05)
        self.declare_parameter('world_frame_suffix', _world_frame_suffix)
        self.declare_parameter('kf_process_noise_acc', _default_process_noise_acc)
        self.declare_parameter('kf_process_noise_vel', _default_process_noise_vel)
        self.declare_parameter('imm_transition_prob', _default_imm_transition_prob)
        self.declare_parameter('kf_measurement_noise', _default_measurement_noise)
        self.declare_parameter('kf_adaptive_cov_max_multiplier', _default_adaptive_cov_max_multiplier)
        self.declare_parameter('kf_initial_covariance', _default_initial_covariance)
        self.declare_parameter('kf_mahalanobis_threshold', _default_mahalanobis_threshold)
        self.declare_parameter('kf_prediction_horizon_sec', 1.0)
        self.declare_parameter('kf_prediction_steps', 25)
        self.declare_parameter('publish_rviz_markers', _default_publish_markers)
        self.declare_parameter('publish_debug_annotated_image', _default_publish_debug_annotated)
        self.declare_parameter('observer1_image_topic', _observer1_image_topic)
        self.declare_parameter('observer2_image_topic', _observer2_image_topic)
        self.declare_parameter('observer1_camera_info_topic', _observer1_camera_info_topic)
        self.declare_parameter('observation_topic_name', _default_observation_topic_name)
        self.declare_parameter('max_tf_delta_ms', _default_max_tf_delta_ms)

        self.observer1_name = str(self.get_parameter('observer1_name').value)
        self.observer2_name = str(self.get_parameter('observer2_name').value)
        self.target_name = str(self.get_parameter('target_name').value)
        self.pose_topic = str(self.get_parameter('pose_topic').value)
        self.odometry_topic = str(self.get_parameter('odometry_topic').value)
        self.estimated_position_topic = str(self.get_parameter('estimated_position_topic').value)
        self.triangulation_raw_topic = str(self.get_parameter('triangulation_raw_topic').value)
        self.error_vector_topic = str(self.get_parameter('error_vector_topic').value)
        self.target_frame_id = str(self.get_parameter('target_frame_id').value)
        self.max_pair_age_ns = int(float(self.get_parameter('max_pair_age_sec').value) * 1_000_000_000.0)
        self.observation_buffer_ns = int(float(self.get_parameter('observation_buffer_sec').value) * 1_000_000_000.0)
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
        self.publish_rviz_markers = bool(self.get_parameter('publish_rviz_markers').value)
        self.publish_debug_annotated_image = bool(self.get_parameter('publish_debug_annotated_image').value)
        self.observer1_image_topic = str(self.get_parameter('observer1_image_topic').value) or f'/{self.observer1_name}/rgb/image_raw'
        self.observer2_image_topic = str(self.get_parameter('observer2_image_topic').value) or f'/{self.observer2_name}/rgb/image_raw'
        self.observation_topic_name = str(self.get_parameter('observation_topic_name').value).strip('/') or 'target_observation'
        self.max_tf_delta_ns = int(float(self.get_parameter('max_tf_delta_ms').value) * 1_000_000.0)

        if self.target_frame_id == 'world':
            self.target_frame_id = f'{self.observer1_name}/{self.world_frame_suffix}'

        self.target_gt_odom_topic = f'/{self.target_name}/{_target_gt_odom_suffix}'
        self.observer1_observation_topic = f'/{self.observer1_name}/{self.observation_topic_name}'
        self.observer2_observation_topic = f'/{self.observer2_name}/{self.observation_topic_name}'
        self.observer1_world_frame = f'{self.observer1_name}/{self.world_frame_suffix}'
        self._observer1_use_compressed = self.observer1_image_topic.endswith('/compressed')
        self.observer1_camera_info_topic = (
            str(self.get_parameter('observer1_camera_info_topic').value).strip()
            or self._camera_info_topic_from_image_topic(
                self.observer1_image_topic,
                self.observer1_name,
            )
        )

        buffer_maxlen = max(32, int(max(1.0, self.processing_rate_hz) * max(1.0, self.observation_buffer_ns / 1e9) * 4.0))
        self.observer1_observations: Deque[SpatialObservation] = collections.deque(maxlen=buffer_maxlen)
        self.observer2_observations: Deque[SpatialObservation] = collections.deque(maxlen=buffer_maxlen)
        self.debug_observations: Dict[str, Deque[SpatialObservation]] = {
            self.observer1_name: collections.deque(maxlen=buffer_maxlen),
            self.observer2_name: collections.deque(maxlen=buffer_maxlen),
        }
        debug_frame_period_ns = int(1_000_000_000.0 / max(1.0, self.processing_rate_hz))
        self.debug_image_match_ns = max(
            5_000_000,
            min(self.max_pair_age_ns, debug_frame_period_ns // 2 if debug_frame_period_ns > 0 else self.max_pair_age_ns),
        )
        self.debug_images: Dict[str, Deque[DebugImageFrame]] = {
            self.observer1_name: collections.deque(maxlen=buffer_maxlen),
        }
        self.debug_matched_pairs: Deque[MatchedObservationPair] = collections.deque(maxlen=buffer_maxlen)
        self.last_debug_publish_key: Dict[str, Tuple[int, int]] = {
            self.observer1_name: (-1, -1),
        }
        self.debug_camera_info: Dict[str, CameraInfo] = {}

        # IMM Transition Probability Matrix (Markov Chain)
        # We define an asymmetric matrix to bias the filter towards Constant Velocity (CV).
        # This rigorously models the prior that drones fly straight most of the time,
        # and accelerations are brief maneuvers.
        # [CV -> CV, CV -> CA]
        # [CA -> CV, CA -> CA]
        p_stay_cv = 0.85  # CV retention: moderate bias toward straight flight
        p_return_to_cv = 0.10  # CA retention: once in CA, stay there to track the full maneuver
        pi = np.array(
            [
                [p_stay_cv, 1.0 - p_stay_cv],
                [p_return_to_cv, 1.0 - p_return_to_cv],
            ],
            dtype=np.float64,
        )
        mu0 = np.array([0.5, 0.5], dtype=np.float64)
        self.imm = IMMFilter(pi, mu0)
        self.imm_initialized = False
        self.imm_last_stamp_ns: Optional[int] = None
        self.filtered_position = np.zeros(3)
        self.filtered_velocity = np.zeros(3)
        self.filtered_acceleration = np.zeros(3)
        self.combined_cov = np.eye(9)
        self.imm_probs = mu0.copy()
        self._imm_cov_trace_reset_thresh = 500.0

        self.target_gt_odom: Optional[Odometry] = None
        self.last_error_log_time = time.time()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.bridge = CvBridge() if self.publish_debug_annotated_image else None

        self.pose_pub = self.create_publisher(PoseStamped, self.pose_topic, 10)
        self.odometry_pub = self.create_publisher(Odometry, self.odometry_topic, 10)
        self.estimated_position_pub = self.create_publisher(Vector3Stamped, self.estimated_position_topic, 10)
        self.triangulation_raw_pub = self.create_publisher(PointStamped, self.triangulation_raw_topic, 10)
        self.error_vector_pub = self.create_publisher(Vector3Stamped, self.error_vector_topic, 10)
        self.predicted_trajectory_pub = self.create_publisher(Path, '/target/predicted_trajectory', 10)
        self.trajectory_markers_pub = self.create_publisher(MarkerArray, '/target/predicted_trajectory_markers', 10)
        self.position_markers_pub = self.create_publisher(MarkerArray, '/target/position_markers', 10)
        self.observer1_debug_annotated_pub = None
        if self.publish_debug_annotated_image:
            self.observer1_debug_annotated_pub = self.create_publisher(
                CompressedImage,
                f'/{self.observer1_name}/triangulation_debug_annotated/compressed',
                10,
            )

        self.create_subscription(Odometry, self.target_gt_odom_topic, self.target_gt_odom_callback, 20)

        _measurement_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        self.observer1_measurement_sub = self.create_subscription(
            UInt8MultiArray,
            self.observer1_observation_topic,
            self.observer1_measurement_callback,
            _measurement_qos,
        )
        self.observer2_measurement_sub = self.create_subscription(
            UInt8MultiArray,
            self.observer2_observation_topic,
            self.observer2_measurement_callback,
            _measurement_qos,
        )

        if self.publish_debug_annotated_image:
            _sensor_qos = QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
            )
            if self._observer1_use_compressed:
                self.observer1_image_sub = self.create_subscription(
                    CompressedImage,
                    self.observer1_image_topic,
                    lambda msg, source=self.observer1_name: self.debug_image_callback(source, msg),
                    _sensor_qos,
                )
            else:
                self.observer1_image_sub = self.create_subscription(
                    Image,
                    self.observer1_image_topic,
                    lambda msg, source=self.observer1_name: self.debug_image_callback(source, msg),
                    _sensor_qos,
                )
            self.observer1_camera_info_sub = self.create_subscription(
                CameraInfo,
                self.observer1_camera_info_topic,
                lambda msg, source=self.observer1_name: self.debug_camera_info_callback(source, msg),
                10,
            )

        self.create_timer(1.0 / self.processing_rate_hz, self.processing_timer_callback)

        self.get_logger().info('Triangulation node initialization complete.')
        self.get_logger().info(f'Observer 1 topic: {self.observer1_observation_topic}')
        self.get_logger().info(f'Observer 2 topic: {self.observer2_observation_topic}')
        self.get_logger().info(f'Observer 1 camera info topic: {self.observer1_camera_info_topic}')
        self.get_logger().info(f'Target frame: {self.target_frame_id}')
        self.get_logger().info(f'Observation buffer: {self.observation_buffer_ns / 1e9:.2f} s')
        if self.publish_debug_annotated_image:
            self.get_logger().info(
                f'Triangulation debug overlay topic: '
                f'/{self.observer1_name}/triangulation_debug_annotated/compressed'
            )

    def target_gt_odom_callback(self, msg: Odometry) -> None:
        self.target_gt_odom = msg

    def observer1_measurement_callback(self, msg: UInt8MultiArray) -> None:
        """Process frozen observer1 measurements published as compact packets."""
        self._handle_measurement_message(self.observer1_name, self.observer1_observations, msg)

    def observer2_measurement_callback(self, msg: UInt8MultiArray) -> None:
        """Process frozen observer2 measurements published as compact packets."""
        self._handle_measurement_message(self.observer2_name, self.observer2_observations, msg)

    @staticmethod
    def _frame_leaf(frame_id: str) -> str:
        return frame_id.rsplit('/', 1)[-1] if frame_id else ''

    @staticmethod
    def _camera_info_topic_from_image_topic(image_topic: str, uav_name: str) -> str:
        if image_topic:
            image_base = image_topic.replace('/compressed', '').rsplit('/image_raw', 1)[0]
            if image_base:
                return f'{image_base}/camera_info'
        return f'/{uav_name}/rgb/camera_info'

    def _shares_common_world_origin(self, frame_id: str) -> bool:
        if not frame_id or not self.target_frame_id:
            return False
        return (
            self._frame_leaf(frame_id) == self.world_frame_suffix
            and self._frame_leaf(self.target_frame_id) == self.world_frame_suffix
        )

    @staticmethod
    def _resolve_camera_frame(source_name: str, camera_frame: str) -> str:
        if camera_frame and source_name not in camera_frame:
            return f'{source_name}/{camera_frame}'
        return camera_frame

    def debug_camera_info_callback(self, source_name: str, msg: CameraInfo) -> None:
        self.debug_camera_info[source_name] = msg

    def debug_image_callback(self, source_name: str, msg) -> None:
        if self.bridge is None:
            return

        if source_name != self.observer1_name:
            return

        camera_info = self.debug_camera_info.get(source_name)
        if camera_info is None:
            return

        try:
            if isinstance(msg, CompressedImage):
                bgr_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
            else:
                bgr_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().warn(
                f'Failed to convert triangulation debug image for {source_name}: {exc}',
                throttle_duration_sec=2.0,
            )
            return

        image_buffer = self.debug_images.get(source_name)
        if image_buffer is None:
            return

        image_stamp_ns = stamp_to_ns(msg.header.stamp)
        image_buffer.append(
            DebugImageFrame(
                stamp_ns=image_stamp_ns,
                bgr_image=bgr_image,
                header=msg.header,
            )
        )
        self._prune_stale_observations(image_stamp_ns)
        self._try_publish_debug_image(source_name, image_stamp_ns)

    def _try_publish_debug_image(self, source_name: str, stamp_ns: int) -> None:
        """Publish a debug overlay only when a local observation and a nearby image frame can be paired."""
        if source_name != self.observer1_name:
            return

        camera_info = self.debug_camera_info.get(source_name)
        if camera_info is None:
            return

        local_obs = self._find_closest_debug_observation(source_name, stamp_ns, self.debug_image_match_ns)
        if local_obs is None:
            return

        debug_frame = self._find_closest_debug_image(source_name, local_obs.stamp_ns, self.debug_image_match_ns)
        if debug_frame is None:
            return

        matched_pair = self._find_closest_matched_pair(local_obs.stamp_ns, self.max_pair_age_ns)
        remote_stamp_ns = matched_pair.remote_obs.stamp_ns if matched_pair is not None else -1
        publish_key = (local_obs.stamp_ns, remote_stamp_ns)
        if self.last_debug_publish_key.get(source_name) == publish_key:
            return

        if self.publish_debug_image(source_name, debug_frame, camera_info, local_obs, matched_pair):
            self.last_debug_publish_key[source_name] = publish_key

    def publish_debug_image(
        self,
        source_name: str,
        debug_frame: DebugImageFrame,
        camera_info: CameraInfo,
        local_obs: SpatialObservation,
        matched_pair: Optional[MatchedObservationPair],
    ) -> bool:
        """Draw the local ray and, when available, the matched remote ray on observer1's image."""
        if source_name != self.observer1_name:
            return False

        debug_pub = self.observer1_debug_annotated_pub
        world_frame = self.observer1_world_frame

        if debug_pub is None:
            return False

        remote_obs = None
        if matched_pair is not None:
            remote_obs = matched_pair.aligned_remote_obs

        camera_frame = self._resolve_camera_frame(source_name, camera_info.header.frame_id)
        try:
            image_time = Time(seconds=debug_frame.header.stamp.sec, nanoseconds=debug_frame.header.stamp.nanosec)
            transform_wc = self.tf_buffer.lookup_transform(
                camera_frame,
                world_frame,
                image_time,
                Duration(seconds=self.tf_timeout_sec),
            )
        except Exception as exc:
            try:
                transform_wc = self.tf_buffer.lookup_transform(
                    camera_frame,
                    world_frame,
                    Time(),
                    Duration(seconds=self.tf_timeout_sec),
                )
            except Exception:
                self.get_logger().warn(
                    f'No transform available for triangulation debug image {source_name}: {exc}',
                    throttle_duration_sec=2.0,
                )
                return False

        rays_to_draw = []
        colors = []
        rays_to_draw.append((local_obs.origin, local_obs.ray))
        colors.append((0, 255, 0))
        if remote_obs is not None:
            rays_to_draw.append((remote_obs.origin, remote_obs.ray))
            colors.append((0, 0, 255))

        if not rays_to_draw:
            return False

        # local_obs and remote_obs are expressed in semantically common world_origin coordinates.
        debug_img = project_ray_on_image(
            debug_frame.bgr_image,
            camera_info,
            transform_wc,
            rays_to_draw,
            color_list=colors,
            length=150.0,
        )

        debug_msg = CompressedImage()
        debug_msg.header = debug_frame.header
        debug_msg.format = 'jpeg'
        debug_msg.data = bytes(cv2.imencode('.jpg', debug_img, [cv2.IMWRITE_JPEG_QUALITY, 75])[1])
        debug_pub.publish(debug_msg)
        return True

    def _find_closest_debug_image(
        self,
        source_name: str,
        stamp_ns: int,
        max_age_ns: int,
    ) -> Optional[DebugImageFrame]:
        buffer = self.debug_images.get(source_name)
        if not buffer:
            return None

        best_frame = None
        best_diff = max_age_ns + 1
        for frame in buffer:
            diff = abs(frame.stamp_ns - stamp_ns)
            if diff < best_diff:
                best_diff = diff
                best_frame = frame

        if best_diff > max_age_ns:
            return None
        return best_frame

    def _find_closest_matched_pair(
        self,
        local_stamp_ns: int,
        max_age_ns: int,
    ) -> Optional[MatchedObservationPair]:
        if not self.debug_matched_pairs:
            return None

        best_pair = None
        best_diff = max_age_ns + 1
        for pair in self.debug_matched_pairs:
            diff = abs(pair.local_obs.stamp_ns - local_stamp_ns)
            if diff < best_diff:
                best_diff = diff
                best_pair = pair

        if best_diff > max_age_ns:
            return None
        return best_pair

    def _find_closest_debug_observation(
        self,
        source_name: str,
        stamp_ns: int,
        max_age_ns: int,
    ) -> Optional[SpatialObservation]:
        buffer = self.debug_observations.get(source_name)
        if not buffer:
            return None

        best_obs = None
        best_diff = max_age_ns + 1
        for obs in buffer:
            diff = abs(obs.stamp_ns - stamp_ns)
            if diff < best_diff:
                best_diff = diff
                best_obs = obs

        if best_diff > max_age_ns:
            return None
        return best_obs

    def _handle_measurement_message(
        self,
        source_name: str,
        buffer: Deque[SpatialObservation],
        msg: UInt8MultiArray,
    ) -> None:
        """Decode one frozen observation packet and push it into the matching buffers."""
        packet = unpack_observation_packet(msg)
        if packet is None:
            self.get_logger().warn(
                f'Ignoring malformed observation packet from {source_name}',
                throttle_duration_sec=2.0,
            )
            return

        stamp_ns, observation_id, confidence, origin, ray_raw, tf_delta_ns, tf_status = packet
        if abs(tf_delta_ns) > self.max_tf_delta_ns:
            self.get_logger().warn(
                f'Ignoring {source_name} observation with TF delta {tf_delta_ns / 1e6:.1f} ms '
                f'(limit {self.max_tf_delta_ns / 1e6:.1f} ms)',
                throttle_duration_sec=2.0,
            )
            return

        try:
            ray = normalize(ray_raw)
        except ValueError:
            self.get_logger().warn(f'Ignoring near-zero ray from {source_name}', throttle_duration_sec=2.0)
            return

        obs = SpatialObservation(
            stamp_ns=stamp_ns,
            origin=origin,
            ray=ray,
            frame_id=f'{source_name}/{self.world_frame_suffix}',
            source_name=source_name,
            observation_id=observation_id,
            confidence=confidence,
            tf_delta_ns=tf_delta_ns,
            tf_status=tf_status,
        )
        obs = self._transform_observation_to_target_frame(obs)
        if obs is None:
            return

        self.debug_observations[source_name].append(obs)
        buffer.append(obs)
        self._prune_stale_observations(stamp_ns)
        self.try_match_and_triangulate()
        self._try_publish_debug_image(source_name, stamp_ns)

    def _transform_observation_to_target_frame(self, obs: SpatialObservation) -> Optional[SpatialObservation]:
        """Express one frozen observation in the triangulation target frame without changing its timestamp."""
        if not obs.frame_id or obs.frame_id == self.target_frame_id:
            return SpatialObservation(
                stamp_ns=obs.stamp_ns,
                origin=obs.origin,
                ray=obs.ray,
                frame_id=self.target_frame_id,
                source_name=obs.source_name,
                observation_id=obs.observation_id,
                confidence=obs.confidence,
                tf_delta_ns=obs.tf_delta_ns,
                tf_status=obs.tf_status,
            )

        if self._shares_common_world_origin(obs.frame_id):
            return SpatialObservation(
                stamp_ns=obs.stamp_ns,
                origin=obs.origin,
                ray=obs.ray,
                frame_id=self.target_frame_id,
                source_name=obs.source_name,
                observation_id=obs.observation_id,
                confidence=obs.confidence,
                tf_delta_ns=obs.tf_delta_ns,
                tf_status=obs.tf_status,
            )

        try:
            sec = obs.stamp_ns // 1_000_000_000
            nanosec = obs.stamp_ns % 1_000_000_000
            lookup_time = Time(seconds=sec, nanoseconds=nanosec)
            transform = self.tf_buffer.lookup_transform(
                self.target_frame_id,
                obs.frame_id,
                lookup_time,
                Duration(seconds=self.tf_timeout_sec),
            )
        except Exception:
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.target_frame_id,
                    obs.frame_id,
                    Time(),
                    Duration(seconds=self.tf_timeout_sec),
                )
            except Exception as exc:
                self.get_logger().warn(
                    f'Cannot transform {obs.source_name} observation from {obs.frame_id} to {self.target_frame_id}: {exc}',
                    throttle_duration_sec=2.0,
                )
                return None

        t = transform.transform.translation
        q = transform.transform.rotation
        rot = quat_to_rotmat(q.x, q.y, q.z, q.w)
        trans = np.array([t.x, t.y, t.z], dtype=np.float64)

        return SpatialObservation(
            stamp_ns=obs.stamp_ns,
            origin=rot @ obs.origin + trans,
            ray=normalize(rot @ obs.ray),
            frame_id=self.target_frame_id,
            source_name=obs.source_name,
            observation_id=obs.observation_id,
            confidence=obs.confidence,
            tf_delta_ns=obs.tf_delta_ns,
            tf_status=obs.tf_status,
        )

    def _prune_stale_observations(self, reference_stamp_ns: Optional[int] = None) -> None:
        """Keep only the time window that can still contribute to matching or debug overlays."""
        if reference_stamp_ns is None:
            reference_stamp_ns = self.get_clock().now().nanoseconds
        cutoff_ns = reference_stamp_ns - self.observation_buffer_ns

        while self.observer1_observations and self.observer1_observations[0].stamp_ns < cutoff_ns:
            self.observer1_observations.popleft()
        while self.observer2_observations and self.observer2_observations[0].stamp_ns < cutoff_ns:
            self.observer2_observations.popleft()
        for debug_buffer in self.debug_observations.values():
            while debug_buffer and debug_buffer[0].stamp_ns < cutoff_ns:
                debug_buffer.popleft()
        for image_buffer in self.debug_images.values():
            while image_buffer and image_buffer[0].stamp_ns < cutoff_ns:
                image_buffer.popleft()
        while self.debug_matched_pairs and self.debug_matched_pairs[0].local_obs.stamp_ns < cutoff_ns:
            self.debug_matched_pairs.popleft()

    def processing_timer_callback(self) -> None:
        now_ns = self.get_clock().now().nanoseconds
        self._prune_stale_observations(now_ns)

        if self.imm_initialized and self.imm_last_stamp_ns is not None:
            dt_since = (now_ns - self.imm_last_stamp_ns) / 1e9
            if dt_since > 0.2:
                self._coast_predict(now_ns)

    def try_match_and_triangulate(self) -> None:
        """Greedily pair the closest timestamps from the two observer buffers and triangulate them."""
        while self.observer1_observations and self.observer2_observations:
            best_obs1 = None
            best_obs2 = None
            best_diff = self.max_pair_age_ns + 1

            for obs1 in self.observer1_observations:
                for obs2 in self.observer2_observations:
                    diff = abs(obs1.stamp_ns - obs2.stamp_ns)
                    if diff < best_diff:
                        best_diff = diff
                        best_obs1 = obs1
                        best_obs2 = obs2

            if best_obs1 is None or best_obs2 is None or best_diff > self.max_pair_age_ns:
                return

            self.observer1_observations.remove(best_obs1)
            self.observer2_observations.remove(best_obs2)

            aligned_obs1, aligned_obs2, aligned_stamp_ns = self._time_align_observations(best_obs1, best_obs2)
            accepted = self.execute_triangulation(
                aligned_obs1.origin,
                aligned_obs1.ray,
                aligned_obs2.origin,
                aligned_obs2.ray,
                aligned_stamp_ns,
            )
            self.debug_matched_pairs.append(
                MatchedObservationPair(
                    local_obs=best_obs1,
                    remote_obs=best_obs2,
                    aligned_local_obs=aligned_obs1,
                    aligned_remote_obs=aligned_obs2,
                    aligned_stamp_ns=aligned_stamp_ns,
                    pair_diff_ns=best_diff,
                    accepted=accepted,
                )
            )
            self._try_publish_debug_image(self.observer1_name, best_obs1.stamp_ns)

    def _time_align_observations(
        self,
        obs1: SpatialObservation,
        obs2: SpatialObservation,
    ) -> Tuple[SpatialObservation, SpatialObservation, int]:
        """Translate both bearing lines to the newer timestamp using the current target motion estimate."""
        target_stamp_ns = max(obs1.stamp_ns, obs2.stamp_ns)
        aligned_obs1 = self._shift_observation_line(obs1, target_stamp_ns)
        aligned_obs2 = self._shift_observation_line(obs2, target_stamp_ns)
        return aligned_obs1, aligned_obs2, target_stamp_ns

    def _shift_observation_line(
        self,
        obs: SpatialObservation,
        target_stamp_ns: int,
    ) -> SpatialObservation:
        origin = obs.origin.copy()
        ray = obs.ray.copy()
        if not self.imm_initialized or self.imm_last_stamp_ns is None:
            return SpatialObservation(
                stamp_ns=target_stamp_ns,
                origin=origin,
                ray=ray,
                frame_id=obs.frame_id,
                source_name=obs.source_name,
                observation_id=obs.observation_id,
                confidence=obs.confidence,
                tf_delta_ns=obs.tf_delta_ns,
                tf_status=obs.tf_status,
            )

        dt = (target_stamp_ns - obs.stamp_ns) / 1e9
        shift = self.filtered_velocity * dt + 0.5 * self.filtered_acceleration * dt * dt
        return SpatialObservation(
            stamp_ns=target_stamp_ns,
            origin=origin + shift,
            ray=ray,
            frame_id=obs.frame_id,
            source_name=obs.source_name,
            observation_id=obs.observation_id,
            confidence=obs.confidence,
            tf_delta_ns=obs.tf_delta_ns,
            tf_status=obs.tf_status,
        )

    def execute_triangulation(
        self,
        origin1: np.ndarray,
        dir1: np.ndarray,
        origin2: np.ndarray,
        dir2: np.ndarray,
        stamp_ns: int,
    ) -> bool:
        """Triangulate one matched pair, reject weak geometry, and feed the accepted point into the IMM."""
        baseline = float(np.linalg.norm(origin2 - origin1))
        if baseline < self.triangulation_min_baseline_m:
            return False

        midpoint, error = triangulate_midpoint(origin1, dir1, origin2, dir2)
        if error > self.max_triangulation_error_m:
            self.get_logger().warn(
                f'Triangulation rejected: ray mismatch {error:.2f} m exceeds {self.max_triangulation_error_m:.2f} m'
            )
            return False

        measurement = midpoint

        raw_msg = PointStamped()
        raw_msg.header.frame_id = self.target_frame_id
        raw_msg.header.stamp.sec = stamp_ns // 1_000_000_000
        raw_msg.header.stamp.nanosec = stamp_ns % 1_000_000_000
        raw_msg.point.x = float(midpoint[0])
        raw_msg.point.y = float(midpoint[1])
        raw_msg.point.z = float(midpoint[2])
        self.triangulation_raw_pub.publish(raw_msg)

        # ---------------------------------------------------------------------
        # Analytic Cartesian Covariance Projection (Information Matrix Method)
        # Instead of feeding the filter a generic spherical noise or an empirical
        # stretched ellipsoid, we compute the exact 3D Cartesian covariance of the 
        # triangulated point using the sum of the Fisher Information Matrices of the rays.
        # This elegantly maps 2D angular uncertainty into a highly anisotropic 3D 
        # ellipsoid that accurately scales with both baseline angle AND distance squared.
        # ---------------------------------------------------------------------
        d1 = float(max(np.linalg.norm(midpoint - origin1), 0.1))
        d2 = float(max(np.linalg.norm(midpoint - origin2), 0.1))
        
        # Convert the Cartesian measurement noise parameter to angular variance.
        # kf_measurement_noise is calibrated as a spatial variance (m²) at some
        # nominal operating distance. We recover the implicit angular variance
        # by dividing by the geometric mean distance squared, so that at the
        # current operating distances the lateral R entries stay consistent
        # with the original parameter intent.
        d_ref = max(np.sqrt(d1 * d2), 0.5)
        angular_var = max(self.kf_measurement_noise / (d_ref ** 2), 1e-8)
        
        # Spatial variance perpendicular to each ray grows with distance squared
        var1 = (d1 ** 2) * angular_var
        var2 = (d2 ** 2) * angular_var
        
        # Information matrices (inverse covariance) for each ray
        # A ray provides no information along its direction (hence I - d*d^T)
        proj1 = np.eye(3) - np.outer(dir1, dir1)
        proj2 = np.eye(3) - np.outer(dir2, dir2)
        info1 = proj1 / var1
        info2 = proj2 / var2
        
        try:
            # The 3D Covariance is the inverse of the Total Information Matrix
            R = np.linalg.inv(info1 + info2)
        except np.linalg.LinAlgError:
            # Fallback for perfectly collinear rays
            R = np.eye(3) * (var1 + var2) * 10.0
        # ---------------------------------------------------------------------

        if not self.imm_initialized:
            self.imm.initialize(measurement, self.kf_initial_covariance)
            self.imm_initialized = True
            self.imm_last_stamp_ns = stamp_ns
            self.combined_cov = np.eye(9) * self.kf_initial_covariance
            self.filtered_position = measurement.copy()
            self.filtered_velocity = np.zeros(3)
            self.filtered_acceleration = np.zeros(3)
        else:
            dt = (stamp_ns - self.imm_last_stamp_ns) / 1_000_000_000.0
            if dt > 0.0:
                Q_cv = build_process_noise_cv(dt, self.kf_process_noise_vel)
                Q_ca = build_process_noise_ca(dt, self.kf_process_noise_acc)
                x_comb, self.combined_cov, self.imm_probs, accepted = self.imm.predict_and_correct(
                    dt,
                    Q_cv,
                    Q_ca,
                    measurement,
                    R,
                    mahalanobis_thresh=self.kf_mahalanobis_threshold,
                )

                if not accepted:
                    self.get_logger().warn(
                        'Outlier measurement rejected by Mahalanobis gating! Filter coasting purely on kinetics.'
                    )

                self.filtered_position = x_comb[:3].copy()
                self.filtered_velocity = x_comb[3:6].copy()
                self.filtered_acceleration = x_comb[6:9].copy()
                self.imm_last_stamp_ns = stamp_ns

                cov_trace = float(np.trace(self.combined_cov))
                if cov_trace > self._imm_cov_trace_reset_thresh:
                    self.get_logger().warn(
                        f'IMM covariance diverged (trace={cov_trace:.1f}), resetting filter at current measurement.'
                    )
                    self.imm.initialize(measurement, self.kf_initial_covariance)
                    self.imm_initialized = True
                    self.combined_cov = np.eye(9) * self.kf_initial_covariance
                    self.filtered_position = measurement.copy()
                    self.filtered_velocity = np.zeros(3)
                    self.filtered_acceleration = np.zeros(3)

        self.publish_pose_estimate(
            self.filtered_position,
            self.filtered_velocity,
            self.filtered_acceleration,
            stamp_ns,
        )
        self.publish_ground_truth_error(self.filtered_position, stamp_ns)
        return True

    def _coast_predict(self, now_ns: int) -> None:
        """Advance the filter without a measurement when the matched observation stream goes temporarily quiet."""
        if self.imm_last_stamp_ns is None:
            return
        dt = (now_ns - self.imm_last_stamp_ns) / 1e9
        if dt <= 0.0 or dt > 5.0:
            return
        self.filtered_position = (
            self.filtered_position
            + self.filtered_velocity * dt
            + 0.5 * self.filtered_acceleration * dt * dt
        )
        self.filtered_velocity = self.filtered_velocity + self.filtered_acceleration * dt
        self.imm_last_stamp_ns = now_ns
        self.get_logger().info(f'Coast predict: dt={dt:.2f}s, no measurement', throttle_duration_sec=2.0)
        self.publish_pose_estimate(
            self.filtered_position,
            self.filtered_velocity,
            self.filtered_acceleration,
            now_ns,
        )

    def publish_pose_estimate(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        acceleration: np.ndarray,
        stamp_ns: int,
    ) -> None:
        """Publish the filtered target state on the pose, odometry, and trajectory topics."""
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

    def _publish_predicted_trajectory(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        acceleration: np.ndarray,
        stamp_ns: int,
    ) -> None:
        sec = int(stamp_ns // 1_000_000_000)
        nanosec = int(stamp_ns % 1_000_000_000)

        path_msg = Path()
        path_msg.header.frame_id = self.target_frame_id
        path_msg.header.stamp.sec = sec
        path_msg.header.stamp.nanosec = nanosec

        dt_step = self.kf_prediction_horizon_sec / self.kf_prediction_steps
        pred_points: List[np.ndarray] = []

        if self.imm_initialized:
            # Start with the currently estimated optimal combined state
            pos = self.filtered_position.copy()
            vel = self.filtered_velocity.copy()
            
            # The effective acceleration is the CA acceleration weighted by the CA model probability
            # (since CV assumes zero acceleration). This perfectly captures the current maneuver intent.
            acc = self.imm.sc_ca.x[6:9].copy() * self.imm.mu[1]

            # Clamp acceleration magnitude to a physically achievable envelope (~1.5g).
            # Any estimate above this limit is measurement-noise leakage, not real physics.
            MAX_ACCEL_PRED = 15.0  # m/s²
            acc_mag = float(np.linalg.norm(acc))
            if acc_mag > MAX_ACCEL_PRED:
                acc = acc * (MAX_ACCEL_PRED / acc_mag)

            # Singer model decay: acceleration decays to zero over time, naturally 
            # modeling the Markov transition back to Constant Velocity (straight flight).
            # A time constant of 0.5s is typical for a drone completing a sharp maneuver.
            tau = 0.5
            alpha = float(np.exp(-dt_step / tau))

            for _ in range(self.kf_prediction_steps + 1):
                pred_points.append(pos.copy())
                pos += vel * dt_step + 0.5 * acc * (dt_step ** 2)
                vel += acc * dt_step
                acc *= alpha
        else:
            for index in range(self.kf_prediction_steps + 1):
                t = index * dt_step
                pred_pos = position + velocity * t + 0.5 * acceleration * (t * t)
                pred_points.append(pred_pos)

        for index, pred_pos in enumerate(pred_points):
            t = index * dt_step
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
        for index in range(n_segments):
            alpha = index / max(1, n_segments - 1)
            marker = Marker()
            marker.header.frame_id = self.target_frame_id
            marker.header.stamp.sec = sec
            marker.header.stamp.nanosec = nanosec
            marker.ns = 'predicted_traj'
            marker.id = index + 1
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.scale.x = 0.06
            marker.scale.y = 0.14
            marker.scale.z = 0.0
            marker.color.r = 1.0
            marker.color.g = float(1.0 - alpha)
            marker.color.b = 0.0
            marker.color.a = float(0.9 - 0.45 * alpha)
            marker.lifetime.sec = 1
            p0 = pred_points[index]
            p1 = pred_points[index + 1]
            marker.points = [
                Point(x=float(p0[0]), y=float(p0[1]), z=float(p0[2])),
                Point(x=float(p1[0]), y=float(p1[1]), z=float(p1[2])),
            ]
            marker_array.markers.append(marker)

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
                % (
                    estimated_position[0],
                    estimated_position[1],
                    estimated_position[2],
                    error[0],
                    error[1],
                    error[2],
                )
            )
            self.last_error_log_time = now


def main() -> None:
    rclpy.init()
    node = TriangulationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()