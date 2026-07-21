#!/usr/bin/env python3

import collections
import json
import math
import os
import struct
import time
import dataclasses
from dataclasses import dataclass
from pathlib import Path as FsPath
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from std_msgs.msg import String, UInt8MultiArray
from tf2_ros import Buffer, StaticTransformBroadcaster, TransformListener
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


def project_ray_on_image(image, camera_info, transform_wc, observations, color_list=None, thickness=2, length=10.0):
    """Project world-frame rays back onto the current camera image for local debug."""
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


# ── Observation packet layout ────────────────────────────────────────────────
# V1 (bearing-only):   [stamp_ns q, obs_id I, conf f, origin 3f, ray 3f, tf_delta q, tf_status B]
# V2 (bearing+range):  V1 fields + [depth_valid B, depth_m f, depth_sigma_m f]
# Triangulation node distinguishes by packet size (len == V1.size → V1, len == V2.size → V2).
OBSERVATION_PACKET_STRUCT = struct.Struct('<qIf3f3fqB')      # V1 — 42 bytes
OBSERVATION_PACKET_V2_STRUCT = struct.Struct('<qIf3f3fqBBff')  # V2 — 51 bytes
TF_STATUS_EXACT = 0
TF_STATUS_FALLBACK_LATEST = 1


def pack_observation_packet(
    image_stamp_ns: int,
    observation_id: int,
    confidence: float,
    camera_origin_world: np.ndarray,
    ray_world: np.ndarray,
    tf_delta_ns: int,
    tf_status: int,
) -> bytes:
    """Pack a V1 (bearing-only) frozen observation into a compact binary payload."""
    return OBSERVATION_PACKET_STRUCT.pack(
        int(image_stamp_ns),
        int(observation_id) & 0xFFFFFFFF,
        float(confidence),
        float(camera_origin_world[0]),
        float(camera_origin_world[1]),
        float(camera_origin_world[2]),
        float(ray_world[0]),
        float(ray_world[1]),
        float(ray_world[2]),
        int(tf_delta_ns),
        int(tf_status),
    )


def pack_observation_packet_v2(
    image_stamp_ns: int,
    observation_id: int,
    confidence: float,
    camera_origin_world: np.ndarray,
    ray_world: np.ndarray,
    tf_delta_ns: int,
    tf_status: int,
    depth_valid: bool,
    depth_m: float,
    depth_sigma_m: float,
) -> bytes:
    """Pack a V2 (bearing + optional range) frozen observation.

    Extends V1 with three depth fields appended at the end so that
    a triangulation node reading only OBSERVATION_PACKET_STRUCT.size
    bytes would still obtain a valid V1 record (forward-compatible slice).

    Fields appended (9 bytes):
        depth_valid   uint8   1 = depth measurement is valid, 0 = bearing-only
        depth_m       float32 measured range along optical ray (metres); 0 if invalid
        depth_sigma_m float32 1-sigma depth noise (metres); 0 if invalid
    """
    return OBSERVATION_PACKET_V2_STRUCT.pack(
        int(image_stamp_ns),
        int(observation_id) & 0xFFFFFFFF,
        float(confidence),
        float(camera_origin_world[0]),
        float(camera_origin_world[1]),
        float(camera_origin_world[2]),
        float(ray_world[0]),
        float(ray_world[1]),
        float(ray_world[2]),
        int(tf_delta_ns),
        int(tf_status),
        int(bool(depth_valid)),
        float(depth_m) if depth_valid else 0.0,
        float(depth_sigma_m) if depth_valid else 0.0,
    )


@dataclass
class DetectionObservation:
    stamp_ns: int
    center_u: float
    center_v: float
    confidence: float
    class_id: int
    class_name: str
    bbox_w: float = 0.0   # bounding box width in pixels
    bbox_h: float = 0.0   # bounding box height in pixels
    source_index: int = -1



@dataclass
class GhostReentryObservation:
    stamp_ns: int
    center_u: float
    center_v: float
    confidence: float


class ImageSpaceKalmanTracker:
    """Image-space Kalman filter for single-target tracking with false positive rejection.

    Implements a 4-state (u, v, u_dot, v_dot) constant-velocity Kalman filter with:
    - Mahalanobis gating for false positive rejection (DeepSORT-inspired)
    - Ego-motion compensation via camera rotation (TAKF Mechanism 1)
    - Distance-adaptive measurement noise using bbox area (TAKF Mechanism 2)
    - Camera-aware covariance transformation (TAKF Mechanism 3)
    - NIS consistency monitoring for persistent false positive detection
    - Joseph-form covariance update for numerical stability

    Pure Python/numpy — no ROS dependencies. All external data (TF, camera intrinsics)
    is passed as numpy arrays by the caller.
    """

    # Track lifecycle states
    UNINITIALIZED = 'UNINITIALIZED'
    TENTATIVE = 'TENTATIVE'
    CONFIRMED = 'CONFIRMED'
    LOST = 'LOST'

    def __init__(
        self,
        chi2_threshold: float = 11.345, # Very permissive for high dynamics
        sigma_a: float = 300.0,         # Process noise suited for fast drone acceleration
        sigma_px: float = 8.0,          # Base measurement noise relaxed
        p_max: float = 800.0,           # Covariance cap increased
        conf_min: float = 0.2,
        conf_beta: float = 1.0,
        n_confirm: int = 3,
        max_age: int = 8,
        nis_window: int = 15,
        nis_threshold: float = 8.0,
        ghost_timeout_sec: float = 0.5,
        # TAKF mechanism parameters
        ego_motion_enabled: bool = True,
        bbox_area_ref: float = 3600.0,
        dist_gamma: float = 0.5,
        dist_ratio_clip: Tuple[float, float] = (0.5, 4.0),
        cam_aware_gating_enabled: bool = True,
    ) -> None:
        # --- Base KF tuning ---
        self._sigma_a = sigma_a
        self._sigma_px = sigma_px
        self._conf_beta = conf_beta
        self._chi2_threshold = chi2_threshold
        self._conf_min = conf_min
        self._n_confirm = n_confirm
        self._max_age = max_age
        self._p_max = p_max
        self._nis_window = nis_window
        self._nis_threshold = nis_threshold

        # --- TAKF mechanisms ---
        self._ego_motion_enabled = ego_motion_enabled
        self._bbox_area_ref = bbox_area_ref
        self._dist_gamma = dist_gamma
        self._dist_ratio_clip = dist_ratio_clip
        self._cam_aware_gating_enabled = cam_aware_gating_enabled

        # Observation matrix: we observe centroid (u, v) directly from the 4-state vector
        self._H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ], dtype=np.float64)

        # Internal state — initialized to defaults; overwritten by initialize()
        self._x: Optional[np.ndarray] = None       # state vector [u, v, u_dot, v_dot]
        self._P: Optional[np.ndarray] = None       # state covariance (4x4)
        self._state: str = self.UNINITIALIZED
        self._prev_stamp_ns: int = 0
        self._consecutive_hits: int = 0
        self._consecutive_misses: int = 0
        self._nis_deque: collections.deque = collections.deque(maxlen=self._nis_window)
        self._predicted_center: Optional[Tuple[float, float]] = None
        self._last_reliable_center: Optional[Tuple[float, float]] = None
        self._last_reliable_stamp_ns: int = 0
        self._last_prediction_guard_reason: Optional[str] = None

        # Ghost memory: last known position after KF reset, used for pre-filtering
        # during UNINITIALIZED state to prevent false positive takeover.
        self._ghost_center: Optional[Tuple[float, float]] = None
        self._ghost_stamp_ns: int = 0
        self._ghost_timeout_ns: int = int(max(0.0, ghost_timeout_sec) * 1_000_000_000)
        self._last_d_ratio: float = 1.0

    # ── Model matrices ───────────────────────────────────────────────

    def _build_F(self, dt: float) -> np.ndarray:
        """State transition matrix for constant-velocity motion model.
        Maps [u, v, u_dot, v_dot] forward by dt seconds."""
        return np.array([
            [1.0, 0.0,  dt, 0.0],
            [0.0, 1.0, 0.0,  dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float64)

    def _build_Q(self, dt: float) -> np.ndarray:
        """Discrete White Noise Acceleration (DWNA) process noise matrix.
        Models unknown target acceleration as white noise with std dev sigma_a.
        The block structure reflects independent u and v acceleration noise."""
        sigma_a_eff = self._sigma_a / max(self._last_d_ratio, 0.3)
        sa2 = sigma_a_eff ** 2
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt
        Q = sa2 * np.array([
            [dt4 / 4.0,       0.0, dt3 / 2.0,       0.0],
            [      0.0, dt4 / 4.0,       0.0, dt3 / 2.0],
            [dt3 / 2.0,       0.0,       dt2,       0.0],
            [      0.0, dt3 / 2.0,       0.0,       dt2],
        ], dtype=np.float64)
        
        # DeepSORT-style proportional baseline noise.
        # At high frame rates (dt=0.01), dt^4 is ~1e-8, making Q_pos nearly zero.
        # We inject a minimum positional uncertainty proportional to the target size.
        approx_bbox_side = np.sqrt(self._bbox_area_ref) / max(self._last_d_ratio, 0.1)
        std_pos = max(0.10 * approx_bbox_side, 1.0)
        Q[0, 0] += std_pos ** 2
        Q[1, 1] += std_pos ** 2
        return Q

    def _build_R(self, detection: DetectionObservation) -> np.ndarray:
        """Distance-adaptive and confidence-dependent measurement noise (TAKF Mechanism 2).

        Bbox area is inversely proportional to squared range: distant targets (small bbox)
        get higher measurement uncertainty, reducing their influence on the state estimate.
        Low-confidence detections also get higher noise. Both effects are multiplicative."""
        A_k = detection.bbox_w * detection.bbox_h
        if A_k > 0.0:
            # Distance ratio: >1 means target is farther than nominal reference distance
            d_ratio = np.sqrt(self._bbox_area_ref / A_k)
            d_ratio = np.clip(d_ratio, self._dist_ratio_clip[0], self._dist_ratio_clip[1])
        else:
            d_ratio = 1.0
        conf = max(detection.confidence, 1e-6)
        # Combined scaling: distance increases noise, low confidence increases noise
        scale = (d_ratio ** (2.0 * self._dist_gamma)) / (conf ** self._conf_beta)
        R_base = (self._sigma_px ** 2) * scale * np.eye(2, dtype=np.float64)
        
        # Proportional bounding box jitter. A large box has higher absolute centroid jitter.
        # 0.10 means 10% of the bbox size is the standard deviation.
        std_u = max(0.10 * detection.bbox_w, 1.0)
        std_v = max(0.10 * detection.bbox_h, 1.0)
        R_prop = np.diag([std_u ** 2, std_v ** 2])
        
        return R_base + R_prop

    def _compute_lambda_cam(self, x: np.ndarray, camera_matrix: Optional[np.ndarray]) -> float:
        """Computes the camera-aware gating threshold multiplier to account for edge distortion."""
        if self._cam_aware_gating_enabled and camera_matrix is not None:
            fx = camera_matrix[0, 0]
            fy = camera_matrix[1, 1]
            cx = camera_matrix[0, 2]
            cy = camera_matrix[1, 2]
            if fx > 10.0 and fy > 10.0:
                x_n = (x[0] - cx) / fx
                y_n = (x[1] - cy) / fy
                return min(1.0 + 0.1 * (x_n ** 2 + y_n ** 2), 1.2)
        return 1.0

    def _remember_reliable_center(self, u: float, v: float, stamp_ns: int) -> None:
        self._last_reliable_center = (float(u), float(v))
        self._last_reliable_stamp_ns = stamp_ns

    def _resolve_frame_size(
        self,
        image_size: Optional[Tuple[int, int]],
        camera_matrix: Optional[np.ndarray],
    ) -> Optional[Tuple[float, float]]:
        if image_size is not None:
            width, height = image_size
            if width > 1 and height > 1:
                return (float(width), float(height))

        if camera_matrix is None:
            return None

        cx = float(camera_matrix[0, 2])
        cy = float(camera_matrix[1, 2])
        if cx > 1.0 and cy > 1.0:
            return (2.0 * cx, 2.0 * cy)

        return None

    def _clip_center_to_frame(
        self,
        center: Tuple[float, float],
        frame_size: Tuple[float, float],
    ) -> Tuple[float, float]:
        width, height = frame_size
        return (
            float(np.clip(center[0], 0.0, max(width - 1.0, 0.0))),
            float(np.clip(center[1], 0.0, max(height - 1.0, 0.0))),
        )

    def _make_recovery_covariance(self) -> np.ndarray:
        p_max_eff = self._p_max / max(self._last_d_ratio ** 2, 0.1)
        return np.diag([p_max_eff, p_max_eff, p_max_eff, p_max_eff]).astype(np.float64)

    def _project_covariance_to_spd(self, covariance: np.ndarray) -> np.ndarray:
        # Keep the covariance symmetric positive definite after the ego-motion Jacobian math.
        covariance_sym = 0.5 * (covariance + covariance.T)
        try:
            eigvals, eigvecs = np.linalg.eigh(covariance_sym)
        except np.linalg.LinAlgError:
            return self._make_recovery_covariance()

        p_max_eff = self._p_max / max(self._last_d_ratio ** 2, 0.1)
        eigvals = np.clip(eigvals, 1e-6, p_max_eff)
        covariance_spd = eigvecs @ np.diag(eigvals) @ eigvecs.T
        return 0.5 * (covariance_spd + covariance_spd.T)

    def _stabilize_prediction(
        self,
        x_pred: np.ndarray,
        P_pred: np.ndarray,
        camera_matrix: Optional[np.ndarray],
        image_size: Optional[Tuple[int, int]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        frame_size = self._resolve_frame_size(image_size, camera_matrix)

        if not np.all(np.isfinite(x_pred)):
            self._last_prediction_guard_reason = 'non-finite predicted state; reverting to last reliable center'
            return self._recover_prediction(x_pred, frame_size)

        if not np.all(np.isfinite(P_pred)):
            x_safe = x_pred.copy()
            if frame_size is not None:
                x_safe[0], x_safe[1] = self._clip_center_to_frame((float(x_safe[0]), float(x_safe[1])), frame_size)
            x_safe[2:4] = 0.0
            self._last_prediction_guard_reason = 'non-finite predicted covariance after ego-motion compensation'
            return x_safe, self._make_recovery_covariance()

        P_safe = self._project_covariance_to_spd(P_pred)
        if frame_size is None:
            return x_pred, P_safe

        width, height = frame_size
        u = float(x_pred[0])
        v = float(x_pred[1])
        far_margin_u = max(0.5 * width, 64.0)
        far_margin_v = max(0.5 * height, 64.0)

        if (
            u < -far_margin_u
            or u > (width - 1.0 + far_margin_u)
            or v < -far_margin_v
            or v > (height - 1.0 + far_margin_v)
        ):
            self._last_prediction_guard_reason = 'predicted center escaped far outside the image; reverting to last reliable center'
            return self._recover_prediction(x_pred, frame_size)

        if u < 0.0 or u > (width - 1.0) or v < 0.0 or v > (height - 1.0):
            x_safe = x_pred.copy()
            x_safe[0], x_safe[1] = self._clip_center_to_frame((u, v), frame_size)
            x_safe[2:4] = 0.0
            self._last_prediction_guard_reason = 'predicted center moved out of frame; clamping to image bounds'
            return x_safe, self._make_recovery_covariance()

        return x_pred, P_safe

    def _recover_prediction(
        self,
        x_pred: np.ndarray,
        frame_size: Optional[Tuple[float, float]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        x_safe = x_pred.copy()
        fallback_center = self._last_reliable_center

        if fallback_center is None:
            if np.all(np.isfinite(x_safe[0:2])):
                fallback_center = (float(x_safe[0]), float(x_safe[1]))
            else:
                fallback_center = (0.0, 0.0)

        if frame_size is not None:
            fallback_center = self._clip_center_to_frame(fallback_center, frame_size)

        x_safe[0] = fallback_center[0]
        x_safe[1] = fallback_center[1]
        x_safe[2:4] = 0.0
        return x_safe, self._make_recovery_covariance()

    # ── Public interface ─────────────────────────────────────────────

    def initialize(self, u: float, v: float, stamp_ns: int, bbox_w: float = 0.0, bbox_h: float = 0.0) -> None:
        """Initialize track to TENTATIVE state.
        Called by ObserverNode when the existing M-of-N gate fires.

        Initial velocity is zero; initial covariance reflects measurement noise for
        position and expected acceleration magnitude for velocity."""
        self._x = np.array([u, v, 0.0, 0.0], dtype=np.float64)
        self._P = np.diag([
            self._sigma_px ** 2,
            self._sigma_px ** 2,
            self._sigma_a ** 2,
            self._sigma_a ** 2,
        ]).astype(np.float64)
        self._state = self.TENTATIVE
        self._prev_stamp_ns = stamp_ns
        # The detection that triggered the M-of-N gate counts as the first hit
        self._consecutive_hits = 1
        self._consecutive_misses = 0
        self._nis_deque.clear()
        self._predicted_center = (u, v)
        self._last_prediction_guard_reason = None
        self._remember_reliable_center(u, v, stamp_ns)
        if bbox_w > 0 and bbox_h > 0:
            A_k = bbox_w * bbox_h
            self._last_d_ratio = float(np.clip(np.sqrt(self._bbox_area_ref / A_k), 0.3, 4.0))
        else:
            self._last_d_ratio = 1.0

    def update(
        self,
        detection: Optional[DetectionObservation],
        stamp_ns: int,
        camera_matrix: Optional[np.ndarray] = None,
        R_ego: Optional[np.ndarray] = None,
        roi_pred_center: Optional[Tuple[float, float]] = None,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[bool, Optional[np.ndarray]]:
        """Run one predict + optional update cycle.

        Args:
            detection:       Best YOLO detection this frame, or None for a miss.
            stamp_ns:        Frame timestamp in nanoseconds.
            camera_matrix:   3x3 intrinsic matrix K from CameraInfo.
            R_ego:           3x3 rotation from previous to current camera frame.
            roi_pred_center: (u, v) TF-based predicted pixel from _get_roi_center().

        Returns:
            (accepted, x_filtered):
                accepted    — True if detection passed the Mahalanobis gate.
                x_filtered  — [u, v, u_dot, v_dot] if state is CONFIRMED, else None.
        """
        if self._state == self.UNINITIALIZED:
            return (False, None)

        self._last_prediction_guard_reason = None

        # ── PREDICT ──────────────────────────────────────────────────
        dt = (stamp_ns - self._prev_stamp_ns) / 1e9
        if dt <= 0.0:
            dt = 1e-3  # guard against non-monotonic or duplicate timestamps

        F = self._build_F(dt)
        Q = self._build_Q(dt)

        # Standard KF prediction: propagate state and covariance forward
        x_pred = F @ self._x
        P_pred = F @ self._P @ F.T + Q

        # ── EGO-MOTION COMPENSATION (TAKF Mechanisms 1 & 3) ──────────
        if self._ego_motion_enabled and R_ego is not None and camera_matrix is not None:
            x_pred, P_pred = self._apply_ego_motion(x_pred, P_pred, R_ego, camera_matrix)

        # ── TF-based prediction override ─────────────────────────────
        # If odometry-based predicted center is available, override the predicted mean.
        # This fuses the TF prediction with KF velocity, adding uncertainty to account
        # for TF prediction inaccuracy.
        if roi_pred_center is not None and self._state in (self.UNINITIALIZED, self.LOST):
            x_pred[0] = roi_pred_center[0]
            x_pred[1] = roi_pred_center[1]
            P_pred[0, 0] += 25.0   # sigma_tf^2 = 5px
            P_pred[1, 1] += 25.0

        x_pred, P_pred = self._stabilize_prediction(x_pred, P_pred, camera_matrix, image_size)

        # Store the ego-motion-corrected predicted center for external use (e.g., ROI)
        self._predicted_center = (float(x_pred[0]), float(x_pred[1]))
        self._prev_stamp_ns = stamp_ns

        # ── HANDLE MISS (no detection this frame) ────────────────────
        if detection is None:
            return self._handle_miss(x_pred, P_pred)

        # Minimum confidence filter: discard very low confidence detections
        if detection.confidence < self._conf_min:
            return self._handle_miss(x_pred, P_pred)

        # ── GATING ───────────────────────────────────────────────────
        R_k = self._build_R(detection)

        # Innovation vector: difference between actual and predicted measurement
        z = np.array([detection.center_u, detection.center_v], dtype=np.float64)
        y = z - self._H @ x_pred

        # Innovation covariance: combines predicted uncertainty and measurement noise
        S = self._H @ P_pred @ self._H.T + R_k

        # Mahalanobis distance squared: measures statistical distance accounting for
        # the shape and orientation of the combined uncertainty ellipse
        S_inv = np.linalg.inv(S)
        d2 = float(y.T @ S_inv @ y)

        # Camera-aware gate scaling (TAKF): dampened to prevent explosion at edges
        lambda_cam = self._compute_lambda_cam(x_pred, camera_matrix)

        # Final gate threshold: base * perspective factor
        # (Confidence already scales R_k, increasing S and reducing d2. Shrinking gamma would wrongly reject true positives when P is large)
        conf = detection.confidence
        gamma = self._chi2_threshold * lambda_cam

        if d2 > gamma:
            # Detection is statistically inconsistent with the predicted track
            return self._handle_miss(x_pred, P_pred, d2=d2, gamma=gamma, conf=conf)

        # ── UPDATE (Joseph form — DeepSORT) ──────────────────────────
        # Kalman gain: optimal weighting between prediction and measurement
        K = P_pred @ self._H.T @ S_inv

        # Posterior state estimate: prediction corrected by weighted innovation
        self._x = x_pred + K @ y

        # Joseph form: P = (I-KH) P (I-KH)^T + K R K^T.
        # Numerically more stable than the standard (I-KH)P form; symmetric by
        # construction and does not accumulate asymmetry from floating-point errors.
        IKH = np.eye(4, dtype=np.float64) - K @ self._H
        self._P = IKH @ P_pred @ IKH.T + K @ R_k @ K.T
        self._P = self._project_covariance_to_spd(self._P)

        # ── NIS monitoring ───────────────────────────────────────────
        # Normalized Innovation Squared: under a correct track, E[NIS] = 2 (measurement
        # DOF). Persistently high NIS indicates the track is following a false positive.
        self._nis_deque.append(d2)

        # ── State machine transitions ────────────────────────────────
        self._consecutive_misses = 0
        self._consecutive_hits += 1

        if self._state == self.TENTATIVE and self._consecutive_hits >= self._n_confirm:
            self._state = self.CONFIRMED

        self._predicted_center = (float(self._x[0]), float(self._x[1]))
        self._remember_reliable_center(float(self._x[0]), float(self._x[1]), stamp_ns)

        if detection is not None and detection.bbox_w > 0 and detection.bbox_h > 0:
            A_k = detection.bbox_w * detection.bbox_h
            self._last_d_ratio = float(np.clip(np.sqrt(self._bbox_area_ref / A_k), 0.3, 4.0))

        # Return filtered state only when CONFIRMED
        x_out = self._x.copy() if self._state == self.CONFIRMED else None
        return (True, x_out)

    def _handle_miss(
        self,
        x_pred: np.ndarray,
        P_pred: np.ndarray,
        d2: float = 0.0,
        gamma: float = 0.0,
        conf: float = 0.0,
    ) -> Tuple[bool, Optional[np.ndarray]]:
        """Common path for frames with no usable detection (miss or gate rejection)."""
        self._consecutive_hits = 0
        self._consecutive_misses += 1
        
        self._x = x_pred
        self._P = P_pred
        
        # Rigorous State Management:
        # 1. Tentative tracks get a short grace period (2 misses) before dying.
        #    Instant-death was too aggressive for edge-of-frame intermittent detections.
        # 2. Confirmed tracks coast using pure kinematics for 'max_age' frames, then go LOST.
        if self._state == self.TENTATIVE:
            if self._consecutive_misses >= 2:
                self._state = self.LOST
        elif self._consecutive_misses >= self._max_age:
            if self._state == self.CONFIRMED:
                self._state = self.LOST
                
        return (False, None)

    def evaluate_gate(
        self,
        center_u: float,
        center_v: float,
        confidence: float,
        bbox_w: float,
        bbox_h: float,
        camera_matrix: Optional[np.ndarray] = None,
    ) -> bool:
        """Evaluate if a generic detection passes the Mahalanobis gate given current state."""
        if self._state == self.UNINITIALIZED or self._x is None or self._P is None:
            return True
            
        if confidence < self._conf_min:
            return False
            
        # Build R matrix using a temporary observation object
        det = DetectionObservation(0, center_u, center_v, confidence, 0, "", bbox_w, bbox_h)
        R_k = self._build_R(det)
        
        z = np.array([center_u, center_v], dtype=np.float64)
        y = z - self._H @ self._x
        S = self._H @ self._P @ self._H.T + R_k
        S_inv = np.linalg.inv(S)
        d2 = float(y.T @ S_inv @ y)
        
        lambda_cam = self._compute_lambda_cam(self._x, camera_matrix)
            
        gamma = self._chi2_threshold * lambda_cam
        return d2 <= gamma

    def _apply_ego_motion(
        self,
        x_pred: np.ndarray,
        P_pred: np.ndarray,
        R_ego: np.ndarray,
        camera_matrix: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compensate the KF prediction for camera rotation between frames (TAKF 1 & 3).

        Step 1: Unproject predicted pixel to a ray in the previous camera frame.
        Step 2: Rotate the ray to the current camera frame using R_ego.
        Step 3: Reproject to pixel coordinates in the current frame.
        Step 4: Rotate the position covariance block through the projection Jacobian
                so the uncertainty ellipse stays geometrically consistent."""
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]

        # Unproject the predicted pixel to an unnormalized ray in the previous camera frame.
        # This moves the 2D prediction into 3D angular space so the camera rotation can
        # be applied before reprojecting to the current frame.
        x_n = (x_pred[0] - cx) / fx
        y_n = (x_pred[1] - cy) / fy
        ray_prev = np.array([x_n, y_n, 1.0], dtype=np.float64)

        # Rotate the ray from the previous to the current camera frame.
        # R_ego = R_cw(k) @ R_wc(k-1) encodes how the camera has rotated between frames.
        ray_curr = R_ego @ ray_prev

        # Guard: a non-positive z-component means the predicted ray points behind the camera.
        # This can happen during aggressive maneuvers. Skip correction and keep the KF prediction.
        if ray_curr[2] <= 1e-6:
            return x_pred, P_pred

        # Reproject the rotated ray to the current camera's pixel coordinates
        u_ego = fx * ray_curr[0] / ray_curr[2] + cx
        v_ego = fy * ray_curr[1] / ray_curr[2] + cy
        x_pred[0] = u_ego
        x_pred[1] = v_ego

        # ── Camera-aware covariance transformation (TAKF Mechanism 3) ──
        # After rotating the predicted mean, rotate the position covariance block
        # through the projection Jacobian so the uncertainty ellipse stays
        # geometrically consistent in the new camera frame.
        x_n_new = (u_ego - cx) / fx
        y_n_new = (v_ego - cy) / fy

        # Projection Jacobian (2x3): maps 3D ray perturbations to pixel perturbations
        J = np.array([
            [fx, 0.0, -fx * x_n_new],
            [0.0, fy, -fy * y_n_new],
        ], dtype=np.float64)

        # Pseudo-inverse of the 2x3 projection Jacobian J.
        # J maps 3D ray perturbations to pixel perturbations; J_inv does the reverse.
        # Used to express the pixel covariance in 3D ray space so R_ego can be applied.
        J_inv = np.linalg.pinv(J, rcond=1e-6)

        # Extract position block from predicted covariance
        P_pos = P_pred[0:2, 0:2].copy()

        # Camera-aware covariance pipeline: pixel -> ray space -> rotate -> pixel space.
        # Ensures the uncertainty ellipse in pixel space reflects both the camera rotation
        # and the perspective geometry at the new predicted pixel location.
        P_ray = J_inv @ P_pos @ J_inv.T
        P_ray_rot = R_ego @ P_ray @ R_ego.T
        P_pos_new = J @ P_ray_rot @ J.T

        # Update position block and position-velocity cross-terms for full consistency
        P_pred[0:2, 0:2] = P_pos_new
        P_pv = P_pred[0:2, 2:4].copy()
        P_pred[0:2, 2:4] = J @ R_ego @ J_inv @ P_pv
        P_pred[2:4, 0:2] = P_pred[0:2, 2:4].T

        return x_pred, P_pred

    def reset(self) -> None:
        """Full reset to UNINITIALIZED state, preserving ghost memory."""
        # Save ghost before clearing — used for pre-filtering during re-acquisition
        if self._last_reliable_center is not None:
            self._ghost_center = self._last_reliable_center
            self._ghost_stamp_ns = self._last_reliable_stamp_ns or self._prev_stamp_ns
        elif self._predicted_center is not None:
            self._ghost_center = self._predicted_center
            self._ghost_stamp_ns = self._prev_stamp_ns
        self._x = None
        self._P = None
        self._state = self.UNINITIALIZED
        self._prev_stamp_ns = 0
        self._consecutive_hits = 0
        self._consecutive_misses = 0
        self._nis_deque.clear()
        self._predicted_center = None

    @property
    def state(self) -> str:
        """Current track state: 'UNINITIALIZED'|'TENTATIVE'|'CONFIRMED'|'LOST'."""
        return self._state

    @property
    def is_confirmed(self) -> bool:
        """True if track is in CONFIRMED state."""
        return self._state == self.CONFIRMED

    @property
    def is_nis_consistent(self) -> bool:
        """True if mean NIS over the sliding window is below nis_threshold.
        Under a correct track, E[NIS] = 2 (measurement DOF). Persistently high NIS
        is the signature of a false positive track on vegetation or infrastructure."""
        if not self._nis_deque:
            return True
        return float(np.mean(self._nis_deque)) <= self._nis_threshold

    @property
    def predicted_center(self) -> Optional[Tuple[float, float]]:
        """Ego-motion-corrected predicted pixel (u, v), or None if unavailable."""
        return self._predicted_center

    @property
    def last_prediction_guard_reason(self) -> Optional[str]:
        """Reason emitted when prediction containment had to override the raw state."""
        return self._last_prediction_guard_reason

    @property
    def ghost_center(self) -> Optional[Tuple[float, float]]:
        """Last known position after KF reset, or None."""
        return self._ghost_center

    def is_ghost_valid(self, stamp_ns: int) -> bool:
        """True if ghost memory is recent enough to use for pre-filtering."""
        if self._ghost_center is None:
            return False
        return (stamp_ns - self._ghost_stamp_ns) < self._ghost_timeout_ns

    def clear_ghost(self) -> None:
        """Explicitly clear ghost memory (e.g. after successful re-acquisition)."""
        self._ghost_center = None
        self._ghost_stamp_ns = 0


class ObserverNode(Node):
    def __init__(self) -> None:
        super().__init__('observer_node')

        _repo_root = FsPath(__file__).resolve().parent.parent
        _config_path = _repo_root / 'config' / 'runtime_stack.yaml'
        _default_device = 'cuda'
        _override_image_topic = ''
        _default_publish_annotated = True
        _default_roi_enabled = False
        _default_roi_reference_imgsz = 320
        _default_roi_margin = 3.0
        _default_roi_min_size = 200
        _default_roi_timeout = 0.5
        _default_roi_full_interval = 10
        _default_roi_init_m = 3
        _default_roi_init_n = 5
        _default_roi_init_spread = 80
        _default_roi_sel_lambda = 0.3
        _default_camera_frame_override = ''
        _default_publish_debug_annotated = False
        _default_publish_detector_evaluation_packets = False
        _default_publish_camera_static_tf = False
        _default_observer_role = ''
        _default_observation_topic_name = 'target_observation'
        _default_allow_tf_fallback = True
        _default_max_tf_delta_ms = 30.0
        # Image-space Kalman filter defaults
        _default_kf_enabled = True
        _default_kf_sigma_a = 200.0
        _default_kf_sigma_px = 6.0
        _default_kf_conf_beta = 1.0
        _default_kf_conf_min = 0.20
        _default_kf_n_confirm = 3
        _default_kf_max_age = 8
        _default_kf_p_max = 400.0
        _default_kf_nis_window = 15
        _default_kf_nis_threshold = 8.0
        _default_kf_ghost_timeout_sec = 0.5
        _default_kf_ghost_reentry_m = 2
        _default_kf_ghost_reentry_n = 3
        _default_kf_ghost_reentry_conf_min = 0.45
        _default_kf_ghost_reentry_spread_px = 0.0
        _default_kf_ego_motion_enabled = True
        _default_kf_bbox_area_ref = 3600.0
        _default_kf_dist_gamma = 0.2
        _default_kf_cam_aware_gating_enabled = False
        
        # Tight Coupling defaults
        _default_tc_enabled = False
        _default_tc_odometry_topic = '/target/odometry_estimate'
        _default_tc_max_3d_age_sec = 0.5
        _default_tc_max_3d_covariance_trace = 15.0
        _default_tc_gating_chi2_threshold = 9.21
        _default_tc_sigma_v_inflate = 1.0  # m/s — gate inflation std-dev over prediction horizon
        _default_tc_r_pixel_sigma_px = 10.0  # [L2-bug2 fix] std-dev in pixels for TC R_pixel floor; defaults to kf_sigma_px

        # Depth Fusion defaults (all inactive by default — zero overhead)
        _default_df_enabled = False
        _default_df_topic_suffix = 'rgb/depth_aligned'
        _default_df_max_align_ms = 40.0
        _default_df_min_range = 0.3
        _default_df_max_range = 8.0
        _default_df_sample_method = 'median'
        _default_df_trimmed_frac = 0.2
        _default_df_min_valid_frac = 0.25
        _default_df_stride = 1
        _default_df_gating = True
        _default_df_gate_k = 3.0
        _default_df_max_odom_age = 0.5
        _default_df_max_odom_cov = 100.0
        _default_df_sigma_base = 0.05
        _default_df_sigma_coeff = 0.02
        # [L3-bug2 fix] Dedicated gate odom topic — independent of tc_enabled
        _default_df_gate_odom_topic = '/target/odometry_estimate'

        # Observer Self-Rejection (Layer 4) defaults — zero overhead when disabled
        _default_osr_enabled = False
        _default_osr_max_odom_age_sec = 0.3        # s:  reject self-detection only with fresh odom
        _default_osr_gate_radius_m = 0.8           # m:  physical drone body radius for pixel gate
        _default_osr_depth_margin_m = 1.5          # m:  foreground margin vs IMM target estimate
        _default_osr_only_if_closer = True          # reject only when observer is in the foreground

        _configured_uav_roles: Dict[str, str] = {}
        _image_topic_by_role: Dict[str, str] = {}
        _camera_info_topic_by_role: Dict[str, str] = {}
        _uav_name = 'uav1'
        _world_frame_suffix = 'world_origin'
        _weights_dir = FsPath(__file__).resolve().parent / 'weights'
        _default_model_path = str(_weights_dir / 'best_openvino_model')
        if _config_path.exists():
            import yaml
            try:
                with open(_config_path, 'r', encoding='utf-8') as _f:
                    _cfg = yaml.safe_load(_f)
                    if _cfg:
                        if 'uav_roles' in _cfg:
                            _configured_uav_roles = {
                                'observer1': _cfg['uav_roles'].get('observer1', '') or '',
                                'observer2': _cfg['uav_roles'].get('observer2', '') or '',
                            }
                            _uav_name = _configured_uav_roles.get('observer2', '') or _uav_name
                        if 'evaluation' in _cfg:
                            _world_frame_suffix = _cfg['evaluation'].get('world_frame_suffix', _world_frame_suffix)
                        if 'vision' in _cfg:
                            _v = _cfg['vision']
                            _default_device = _v.get('device', _default_device)
                            _cfg_model_path = str(_v.get('model_path', '') or '').strip()
                            if _cfg_model_path:
                                _default_model_path = _cfg_model_path
                            _image_topic_by_role = {
                                'observer1': _v.get('main_image_topic', '') or '',
                                'observer2': _v.get('observer_image_topic', '') or '',
                            }
                            _camera_info_topic_by_role = {
                                'observer1': _v.get('main_camera_info_topic', '') or '',
                                'observer2': _v.get('observer_camera_info_topic', '') or '',
                            }
                            _override_image_topic = _image_topic_by_role.get('observer2', '') or ''
                            _default_publish_annotated = bool(_v.get('publish_annotated_image', _default_publish_annotated))
                            _default_publish_debug_annotated = bool(
                                _v.get('publish_debug_annotated_image', _default_publish_debug_annotated)
                            )
                            _default_publish_detector_evaluation_packets = bool(
                                _v.get(
                                    'publish_detector_evaluation_packets',
                                    _default_publish_detector_evaluation_packets,
                                )
                            )
                            _default_publish_camera_static_tf = bool(
                                _v.get('publish_camera_static_tf', _default_publish_camera_static_tf)
                            )
                            _default_roi_enabled = bool(_v.get('roi_crop_enabled', _default_roi_enabled))
                            _default_roi_reference_imgsz = int(
                                _v.get('roi_reference_imgsz', _v.get('roi_crop_imgsz', _default_roi_reference_imgsz))
                            )
                            _default_roi_margin = float(_v.get('roi_margin_multiplier', _default_roi_margin))
                            _default_roi_min_size = int(_v.get('roi_min_size_px', _default_roi_min_size))
                            _default_roi_timeout = float(_v.get('roi_track_timeout_sec', _default_roi_timeout))
                            _default_roi_full_interval = int(_v.get('roi_full_scan_interval', _default_roi_full_interval))
                            _default_roi_init_m = int(_v.get('roi_init_m', _default_roi_init_m))
                            _default_roi_init_n = int(_v.get('roi_init_n', _default_roi_init_n))
                            _default_roi_init_spread = int(_v.get('roi_init_max_spread_px', _default_roi_init_spread))
                            _default_roi_sel_lambda = float(_v.get('roi_selection_lambda', _default_roi_sel_lambda))
                            _default_camera_frame_override = _v.get('camera_frame_override', '') or ''
                            _default_observation_topic_name = (
                                _v.get('observation_topic_name', _default_observation_topic_name) or _default_observation_topic_name
                            )
                            _default_allow_tf_fallback = bool(
                                _v.get('allow_tf_fallback', _default_allow_tf_fallback)
                            )
                            _default_max_tf_delta_ms = float(
                                _v.get('max_tf_delta_ms', _default_max_tf_delta_ms)
                            )
                        # Read image-space KF config from YAML
                        if 'kalman_filter' in _cfg:
                            _kf = _cfg['kalman_filter']
                            _default_kf_enabled = bool(_kf.get('kf_enabled', _default_kf_enabled))
                            _default_kf_sigma_a = float(_kf.get('kf_sigma_a', _default_kf_sigma_a))
                            _default_kf_sigma_px = float(_kf.get('kf_sigma_px', _default_kf_sigma_px))
                            _default_kf_conf_beta = float(_kf.get('kf_conf_beta', _default_kf_conf_beta))
                            _default_kf_conf_min = float(_kf.get('kf_conf_min', _default_kf_conf_min))
                            _default_kf_n_confirm = int(_kf.get('kf_n_confirm', _default_kf_n_confirm))
                            _default_kf_max_age = int(_kf.get('kf_max_age', _default_kf_max_age))
                            _default_kf_p_max = float(_kf.get('kf_p_max', _default_kf_p_max))
                            _default_kf_nis_window = int(_kf.get('kf_nis_window', _default_kf_nis_window))
                            _default_kf_nis_threshold = float(_kf.get('kf_nis_threshold', _default_kf_nis_threshold))
                            _default_kf_ghost_timeout_sec = float(
                                _kf.get('kf_ghost_timeout_sec', _default_kf_ghost_timeout_sec)
                            )
                            _default_kf_ghost_reentry_m = int(
                                _kf.get('kf_ghost_reentry_m', _default_kf_ghost_reentry_m)
                            )
                            _default_kf_ghost_reentry_n = int(
                                _kf.get('kf_ghost_reentry_n', _default_kf_ghost_reentry_n)
                            )
                            _default_kf_ghost_reentry_conf_min = float(
                                _kf.get('kf_ghost_reentry_conf_min', _default_kf_ghost_reentry_conf_min)
                            )
                            _default_kf_ghost_reentry_spread_px = float(
                                _kf.get('kf_ghost_reentry_spread_px', _default_kf_ghost_reentry_spread_px)
                            )
                            _default_kf_ego_motion_enabled = bool(_kf.get('kf_ego_motion_enabled', _default_kf_ego_motion_enabled))
                            _default_kf_bbox_area_ref = float(_kf.get('kf_bbox_area_ref', _default_kf_bbox_area_ref))
                            _default_kf_dist_gamma = float(_kf.get('kf_dist_gamma', _default_kf_dist_gamma))
                            _default_kf_cam_aware_gating_enabled = bool(_kf.get('kf_cam_aware_gating_enabled', _default_kf_cam_aware_gating_enabled))
                        
                         # Read Tight Coupling config from YAML
                        if 'tight_coupling' in _cfg:
                            _tc = _cfg['tight_coupling']
                            _default_tc_enabled = bool(_tc.get('enabled', _default_tc_enabled))
                            _default_tc_odometry_topic = str(_tc.get('target_odometry_topic', _default_tc_odometry_topic))
                            _default_tc_max_3d_age_sec = float(_tc.get('max_3d_age_sec', _default_tc_max_3d_age_sec))
                            _default_tc_max_3d_covariance_trace = float(_tc.get('max_3d_covariance_trace', _default_tc_max_3d_covariance_trace))
                            _default_tc_gating_chi2_threshold = float(_tc.get('gating_chi2_threshold', _default_tc_gating_chi2_threshold))
                            _default_tc_sigma_v_inflate = float(_tc.get('sigma_v_inflate', _default_tc_sigma_v_inflate))
                            # [L2-bug2] Read parametric R_pixel sigma; falls back to kf_sigma_px for consistency
                            _default_tc_r_pixel_sigma_px = float(
                                _tc.get('r_pixel_sigma_px', _default_kf_sigma_px)
                            )

                        # Read Depth Fusion config from YAML
                        if 'depth_fusion' in _cfg:
                            _df = _cfg['depth_fusion']
                            _default_df_enabled = bool(_df.get('enabled', _default_df_enabled))
                            _default_df_topic_suffix = str(_df.get('depth_topic_suffix', _default_df_topic_suffix))
                            _default_df_max_align_ms = float(_df.get('depth_max_align_ms', _default_df_max_align_ms))
                            _default_df_min_range = float(_df.get('min_range_m', _default_df_min_range))
                            _default_df_max_range = float(_df.get('max_range_m', _default_df_max_range))
                            _default_df_sample_method = str(_df.get('sample_method', _default_df_sample_method))
                            _default_df_trimmed_frac = float(_df.get('trimmed_mean_fraction', _default_df_trimmed_frac))
                            _default_df_min_valid_frac = float(_df.get('min_valid_pixel_fraction', _default_df_min_valid_frac))
                            _default_df_stride = int(_df.get('depth_sample_stride', _default_df_stride))
                            _default_df_gating = bool(_df.get('gating_enabled', _default_df_gating))
                            _default_df_gate_k = float(_df.get('gate_k', _default_df_gate_k))
                            _default_df_max_odom_age = float(_df.get('max_odom_age_sec', _default_df_max_odom_age))
                            _default_df_max_odom_cov = float(_df.get('max_odom_covariance_trace', _default_df_max_odom_cov))
                            _default_df_sigma_base = float(_df.get('sigma_z_base_m', _default_df_sigma_base))
                            _default_df_sigma_coeff = float(_df.get('sigma_z_range_coeff', _default_df_sigma_coeff))
                            # [L3-bug2] Allow overriding the depth gate odom topic
                            _default_df_gate_odom_topic = str(
                                _df.get('gate_odom_topic', _default_df_gate_odom_topic)
                            )
                        if 'observer_self_rejection' in _cfg:
                            _osr = _cfg['observer_self_rejection']
                            _default_osr_enabled = bool(_osr.get('enabled', _default_osr_enabled))
                            _default_osr_max_odom_age_sec = float(
                                _osr.get('max_odom_age_sec', _default_osr_max_odom_age_sec)
                            )
                            _default_osr_gate_radius_m = float(
                                _osr.get('rejection_gate_radius_m', _default_osr_gate_radius_m)
                            )
                            _default_osr_depth_margin_m = float(
                                _osr.get('rejection_depth_margin_m', _default_osr_depth_margin_m)
                            )
                            _default_osr_only_if_closer = bool(
                                _osr.get('rejection_only_if_closer', _default_osr_only_if_closer)
                            )
            except Exception:
                pass

        self.declare_parameter('uav_name', _uav_name)
        self.declare_parameter('observer_role', _default_observer_role)
        self.declare_parameter('model_path', _default_model_path)
        self.declare_parameter('confidence_threshold', 0.25)
        self.declare_parameter('iou_threshold', 0.45)
        self.declare_parameter('imgsz', 896)
        self.declare_parameter('max_detections', 100)
        self.declare_parameter('device', _default_device)
        self.declare_parameter('line_width', 2)
        self.declare_parameter('target_class_id', -1)
        self.declare_parameter('processing_rate_hz', 100.0)
        self.declare_parameter('tf_timeout_sec', 0.05)
        self.declare_parameter('world_frame_suffix', _world_frame_suffix)
        self.declare_parameter('publish_annotated_image', _default_publish_annotated)
        self.declare_parameter('publish_debug_annotated_image', _default_publish_debug_annotated)
        self.declare_parameter('publish_detector_evaluation_packets', _default_publish_detector_evaluation_packets)
        self.declare_parameter('publish_camera_static_tf', _default_publish_camera_static_tf)
        self.declare_parameter('roi_crop_enabled', _default_roi_enabled)
        self.declare_parameter('roi_reference_imgsz', _default_roi_reference_imgsz)
        self.declare_parameter('roi_crop_imgsz', _default_roi_reference_imgsz)
        self.declare_parameter('roi_margin_multiplier', _default_roi_margin)
        self.declare_parameter('roi_min_size_px', _default_roi_min_size)
        self.declare_parameter('roi_track_timeout_sec', _default_roi_timeout)
        self.declare_parameter('roi_full_scan_interval', _default_roi_full_interval)
        self.declare_parameter('roi_init_m', _default_roi_init_m)
        self.declare_parameter('roi_init_n', _default_roi_init_n)
        self.declare_parameter('roi_init_max_spread_px', _default_roi_init_spread)
        self.declare_parameter('roi_selection_lambda', _default_roi_sel_lambda)
        self.declare_parameter('camera_frame_override', _default_camera_frame_override)
        self.declare_parameter('camera_info_topic', '')
        self.declare_parameter('observation_topic_name', _default_observation_topic_name)
        self.declare_parameter('allow_tf_fallback', _default_allow_tf_fallback)
        self.declare_parameter('max_tf_delta_ms', _default_max_tf_delta_ms)
        # Image-space Kalman filter parameters
        self.declare_parameter('kf_enabled', _default_kf_enabled)
        self.declare_parameter('kf_sigma_a', _default_kf_sigma_a)
        self.declare_parameter('kf_sigma_px', _default_kf_sigma_px)
        self.declare_parameter('kf_conf_beta', _default_kf_conf_beta)
        self.declare_parameter('kf_conf_min', _default_kf_conf_min)
        self.declare_parameter('kf_n_confirm', _default_kf_n_confirm)
        self.declare_parameter('kf_max_age', _default_kf_max_age)
        self.declare_parameter('kf_p_max', _default_kf_p_max)
        self.declare_parameter('kf_nis_window', _default_kf_nis_window)
        self.declare_parameter('kf_nis_threshold', _default_kf_nis_threshold)
        self.declare_parameter('kf_ego_motion_enabled', _default_kf_ego_motion_enabled)
        self.declare_parameter('kf_ghost_timeout_sec', _default_kf_ghost_timeout_sec)
        self.declare_parameter('kf_ghost_reentry_m', _default_kf_ghost_reentry_m)
        self.declare_parameter('kf_ghost_reentry_n', _default_kf_ghost_reentry_n)
        self.declare_parameter('kf_ghost_reentry_conf_min', _default_kf_ghost_reentry_conf_min)
        self.declare_parameter('kf_ghost_reentry_spread_px', _default_kf_ghost_reentry_spread_px)
        self.declare_parameter('kf_bbox_area_ref', _default_kf_bbox_area_ref)
        self.declare_parameter('kf_dist_gamma', _default_kf_dist_gamma)
        self.declare_parameter('kf_cam_aware_gating_enabled', _default_kf_cam_aware_gating_enabled)

        # Tight Coupling parameters
        self.declare_parameter('tc_enabled', _default_tc_enabled)
        self.declare_parameter('tc_odometry_topic', _default_tc_odometry_topic)
        self.declare_parameter('tc_max_3d_age_sec', _default_tc_max_3d_age_sec)
        self.declare_parameter('tc_max_3d_covariance_trace', _default_tc_max_3d_covariance_trace)
        self.declare_parameter('tc_gating_chi2_threshold', _default_tc_gating_chi2_threshold)
        self.declare_parameter('tc_sigma_v_inflate', _default_tc_sigma_v_inflate)
        self.declare_parameter('tc_r_pixel_sigma_px', _default_tc_r_pixel_sigma_px)

        # Observer Self-Rejection (Layer 4) parameters
        self.declare_parameter('osr_enabled', _default_osr_enabled)
        self.declare_parameter('osr_max_odom_age_sec', _default_osr_max_odom_age_sec)
        self.declare_parameter('osr_gate_radius_m', _default_osr_gate_radius_m)
        self.declare_parameter('osr_depth_margin_m', _default_osr_depth_margin_m)
        self.declare_parameter('osr_only_if_closer', _default_osr_only_if_closer)

        self.uav_name = str(self.get_parameter('uav_name').value)
        _requested_observer_role = str(self.get_parameter('observer_role').value).strip()
        _matching_observer_roles = [
            role_name
            for role_name in ('observer1', 'observer2')
            if self.uav_name and _configured_uav_roles.get(role_name, '') == self.uav_name
        ]
        self.observer_role = self._resolve_observer_role(_requested_observer_role, _matching_observer_roles)
        self.model_path = os.path.expanduser(str(self.get_parameter('model_path').value)).strip()
        if self.model_path and not os.path.isabs(self.model_path):
            self.model_path = str((_repo_root / self.model_path).resolve())
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
        self.publish_debug_annotated_image = bool(self.get_parameter('publish_debug_annotated_image').value)
        self.publish_detector_evaluation_packets = bool(
            self.get_parameter('publish_detector_evaluation_packets').value
        )
        self.publish_camera_static_tf = bool(self.get_parameter('publish_camera_static_tf').value)
        self.roi_crop_enabled = bool(self.get_parameter('roi_crop_enabled').value)
        self.roi_reference_imgsz = int(self.get_parameter('roi_reference_imgsz').value)
        _legacy_roi_crop_imgsz = int(self.get_parameter('roi_crop_imgsz').value)
        if (
            _legacy_roi_crop_imgsz != _default_roi_reference_imgsz
            and self.roi_reference_imgsz == _default_roi_reference_imgsz
        ):
            self.roi_reference_imgsz = _legacy_roi_crop_imgsz
        elif (
            _legacy_roi_crop_imgsz != _default_roi_reference_imgsz
            and self.roi_reference_imgsz != _legacy_roi_crop_imgsz
        ):
            self.get_logger().warn(
                'Both roi_reference_imgsz and legacy roi_crop_imgsz are set with different values. '
                'Using roi_reference_imgsz.',
            )
        self.roi_margin = float(self.get_parameter('roi_margin_multiplier').value)
        self.roi_min_size = int(self.get_parameter('roi_min_size_px').value)
        self.roi_timeout_ns = int(float(self.get_parameter('roi_track_timeout_sec').value) * 1_000_000_000)
        self.roi_full_scan_interval = max(1, int(self.get_parameter('roi_full_scan_interval').value))
        self.roi_init_m = max(1, int(self.get_parameter('roi_init_m').value))
        self.roi_init_n = max(self.roi_init_m, int(self.get_parameter('roi_init_n').value))
        self.roi_init_spread = float(self.get_parameter('roi_init_max_spread_px').value)
        self.roi_sel_lambda = float(self.get_parameter('roi_selection_lambda').value)
        self._camera_frame_override = str(self.get_parameter('camera_frame_override').value) or ''
        self.observation_topic_name = str(self.get_parameter('observation_topic_name').value).strip('/') or 'target_observation'
        self.allow_tf_fallback = bool(self.get_parameter('allow_tf_fallback').value)
        self.max_tf_delta_ns = int(float(self.get_parameter('max_tf_delta_ms').value) * 1_000_000.0)
        self.kf_ghost_timeout_sec = max(0.0, float(self.get_parameter('kf_ghost_timeout_sec').value))
        self.kf_ghost_reentry_m = max(1, int(self.get_parameter('kf_ghost_reentry_m').value))
        self.kf_ghost_reentry_n = max(self.kf_ghost_reentry_m, int(self.get_parameter('kf_ghost_reentry_n').value))
        self.kf_ghost_reentry_conf_min = max(
            float(self.get_parameter('kf_conf_min').value),
            float(self.get_parameter('kf_ghost_reentry_conf_min').value),
        )
        self.kf_ghost_reentry_spread_px = max(
            0.0,
            float(self.get_parameter('kf_ghost_reentry_spread_px').value),
        )

        # Tight Coupling logic
        self.tc_enabled = bool(self.get_parameter('tc_enabled').value)
        self.tc_odometry_topic = str(self.get_parameter('tc_odometry_topic').value)
        self.tc_max_3d_age_ns = int(float(self.get_parameter('tc_max_3d_age_sec').value) * 1_000_000_000)
        self.tc_max_3d_covariance_trace = float(self.get_parameter('tc_max_3d_covariance_trace').value)
        self.tc_gating_chi2_threshold = float(self.get_parameter('tc_gating_chi2_threshold').value)
        self.tc_sigma_v_inflate = float(self.get_parameter('tc_sigma_v_inflate').value)
        # [L2-bug2 fix] Parametric R_pixel variance for TC gate; squared here once for efficiency
        _tc_r_px_sigma = float(self.get_parameter('tc_r_pixel_sigma_px').value)
        self.tc_r_pixel_variance: float = _tc_r_px_sigma ** 2
        
        self._tc_latest_odom: Optional[Odometry] = None
        self._tc_latest_odom_stamp_ns: int = 0
        # [L2-bug1 fix] Acceleration estimate for velocity-compensated TC projection.
        # Computed via first-order backward difference of twist.linear filtered with EMA.
        # State: previous velocity sample and its timestamp.
        self._tc_prev_vel: Optional[np.ndarray] = None
        self._tc_prev_vel_stamp_ns: int = 0
        self._tc_accel_estimate: np.ndarray = np.zeros(3, dtype=np.float64)
        self._tc_accel_ema_alpha: float = 0.3  # EMA smoothing factor (lower = more smoothing)

        # Depth Fusion logic — zero overhead when disabled
        self.df_enabled: bool = _default_df_enabled
        self.df_topic_suffix: str = _default_df_topic_suffix
        self.df_max_align_ns: int = int(_default_df_max_align_ms * 1_000_000)
        self.df_min_range: float = _default_df_min_range
        self.df_max_range: float = _default_df_max_range
        self.df_sample_method: str = _default_df_sample_method
        self.df_trimmed_frac: float = _default_df_trimmed_frac
        self.df_min_valid_frac: float = _default_df_min_valid_frac
        self.df_stride: int = max(1, _default_df_stride)
        self.df_gating: bool = _default_df_gating
        self.df_gate_k: float = _default_df_gate_k
        self.df_max_odom_age_ns: int = int(_default_df_max_odom_age * 1_000_000_000)
        self.df_max_odom_cov: float = _default_df_max_odom_cov
        self.df_sigma_base: float = _default_df_sigma_base
        self.df_sigma_coeff: float = _default_df_sigma_coeff
        # Depth frame cache: deque of (stamp_ns, np.ndarray) tuples, maxlen keeps last ~10 frames
        self._depth_cache: collections.deque = collections.deque(maxlen=10)
        # [L3-bug2 fix] Dedicated IMM odometry state for the depth consistency gate.
        # Populated by _df_gate_odom_callback; independent of _tc_latest_odom so that
        # the gate works even when tc_enabled=False.
        self._df_gate_odom: Optional[Odometry] = None
        self._df_gate_odom_stamp_ns: int = 0
        self._df_gate_odom_topic: str = _default_df_gate_odom_topic

        # Observer Self-Rejection (Layer 4) state — all zero-overhead when osr_enabled=False
        self.osr_enabled: bool = bool(self.get_parameter('osr_enabled').value)
        self.osr_max_odom_age_ns: int = int(
            float(self.get_parameter('osr_max_odom_age_sec').value) * 1_000_000_000
        )
        self.osr_gate_radius_m: float = float(self.get_parameter('osr_gate_radius_m').value)
        self.osr_depth_margin_m: float = float(self.get_parameter('osr_depth_margin_m').value)
        self.osr_only_if_closer: bool = bool(self.get_parameter('osr_only_if_closer').value)
        # Caches the other observer's odometry; populated only when osr_enabled=True
        self._other_observer_odom: Optional[Odometry] = None
        self._other_observer_odom_stamp_ns: int = 0
        # Holds the last projected pixel of the other observer for debug overlay
        self._last_other_observer_proj: Optional[Tuple[float, float]] = None

        if _requested_observer_role and _requested_observer_role not in ('observer1', 'observer2'):
            self.get_logger().warn(
                f'Invalid observer_role={_requested_observer_role!r}; expected observer1 or observer2. '
                f'Using resolved role {self.observer_role}.',
            )
        elif not _requested_observer_role and len(_matching_observer_roles) > 1:
            self.get_logger().warn(
                f'Ambiguous observer role for uav_name={self.uav_name}: matched {", ".join(_matching_observer_roles)}. '
                'Pass observer_role explicitly from the launcher. Falling back to observer2 config.',
            )
        elif not _requested_observer_role and not _matching_observer_roles:
            self.get_logger().warn(
                f'Could not infer observer_role from uav_name={self.uav_name}. '
                'Pass observer_role explicitly from the launcher. Falling back to observer2 config.',
            )

        self.image_topic = (
            _image_topic_by_role.get(self.observer_role, '')
            or _override_image_topic
            or f'/{self.uav_name}/rgb/image_raw'
        )
        self._use_compressed = self.image_topic.endswith('/compressed')
        _img_base = self.image_topic.replace('/compressed', '').rsplit('/image_raw', 1)[0]
        _camera_info_topic_override = str(self.get_parameter('camera_info_topic').value).strip()
        self.camera_info_topic = (
            _camera_info_topic_override
            or _camera_info_topic_by_role.get(self.observer_role, '')
            or f'{_img_base}/camera_info'
        )
        self.detection_topic = f'/{self.uav_name}/detection3'
        self.detector_evaluation_topic = f'/{self.uav_name}/detector_evaluation'
        self.observation_topic = f'/{self.uav_name}/{self.observation_topic_name}'
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
        self._log_infer_total_ms = 0.0
        self._log_infer_max_ms = 0.0
        self._log_roi_frames = 0
        self._log_full_frames = 0
        self._log_annotated_publish_count = 0
        self.last_log_time = time.time()
        self._last_image_time = time.time()

        self._roi_last_bbox: Optional[Tuple[float, float, float, float]] = None
        self._roi_last_center: Optional[Tuple[float, float]] = None
        self._roi_last_stamp_ns: int = 0
        self._roi_frame_counter: int = 0
        self._roi_vel_u: float = 0.0
        self._roi_vel_v: float = 0.0
        self._roi_consecutive_misses: int = 0
        self._init_gate_buffer: collections.deque = collections.deque(maxlen=self.roi_init_n)
        self._ghost_reentry_buffer: collections.deque = collections.deque(maxlen=self.kf_ghost_reentry_n)
        self._ghost_reentry_ready: bool = False

        self._target_odom: Optional[Odometry] = None
        self._target_odom_stamp_ns: int = 0
        self._observation_id_counter: int = 0

        # ── Image-space Kalman filter ─────────────────────────────────
        self.kf_enabled = bool(self.get_parameter('kf_enabled').value)
        self._kf_tracker_prev_stamp_ns: int = 0
        self._kf_tracker = ImageSpaceKalmanTracker(
            sigma_a=float(self.get_parameter('kf_sigma_a').value),
            sigma_px=float(self.get_parameter('kf_sigma_px').value),
            conf_beta=float(self.get_parameter('kf_conf_beta').value),
            conf_min=float(self.get_parameter('kf_conf_min').value),
            n_confirm=int(self.get_parameter('kf_n_confirm').value),
            max_age=int(self.get_parameter('kf_max_age').value),
            p_max=float(self.get_parameter('kf_p_max').value),
            nis_window=int(self.get_parameter('kf_nis_window').value),
            nis_threshold=float(self.get_parameter('kf_nis_threshold').value),
            ghost_timeout_sec=self.kf_ghost_timeout_sec,
            ego_motion_enabled=bool(self.get_parameter('kf_ego_motion_enabled').value),
            bbox_area_ref=float(self.get_parameter('kf_bbox_area_ref').value),
            dist_gamma=float(self.get_parameter('kf_dist_gamma').value),
            cam_aware_gating_enabled=bool(self.get_parameter('kf_cam_aware_gating_enabled').value),
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self._static_tf_broadcaster = None
        self._yaw_offset = 0.0
        if self.publish_camera_static_tf:
            self._publish_camera_static_tf(_config_path)

        self.detection_pub = self.create_publisher(CompressedImage, self.detection_topic + '/compressed', 10)
        self.debug_annotated_pub = None
        if self.publish_debug_annotated_image:
            self.debug_annotated_pub = self.create_publisher(
                CompressedImage,
                f'/{self.uav_name}/debug_annotated/compressed',
                10,
            )
        self.detector_evaluation_pub = None
        if self.publish_detector_evaluation_packets:
            self.detector_evaluation_pub = self.create_publisher(String, self.detector_evaluation_topic, 10)
        _sensor_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        _measurement_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        self.observation_pub = self.create_publisher(UInt8MultiArray, self.observation_topic, _measurement_qos)
        if self._use_compressed:
            self.create_subscription(CompressedImage, self.image_topic, self.image_callback, _sensor_qos)
        else:
            self.create_subscription(Image, self.image_topic, self.image_callback, _sensor_qos)
        self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_callback, 10)
        self.create_subscription(Odometry, '/target/odometry_estimate', self._target_odom_callback, 10)
        
        if self.tc_enabled:
            self.create_subscription(Odometry, self.tc_odometry_topic, self._tc_odom_callback, 10)

        # Depth subscriber — ONLY created when depth_fusion.enabled=true.
        # When disabled this block is never executed, so there is absolutely
        # zero subscription, decode, or cache overhead.
        if self.df_enabled:
            _depth_topic = f'/{self.uav_name}/{self.df_topic_suffix}'
            _depth_qos = QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
            )
            self.create_subscription(Image, _depth_topic, self._depth_callback, _depth_qos)
            self.get_logger().info(f'Depth fusion ENABLED — subscribing to: {_depth_topic}')
            # [L3-bug2 fix] Subscribe to IMM odometry for the depth gate, independent
            # of tc_enabled.  This ensures _check_depth_gate always has a 3D reference.
            if self.df_gating:
                self.create_subscription(
                    Odometry,
                    self._df_gate_odom_topic,
                    self._df_gate_odom_callback,
                    10,
                )
                self.get_logger().info(
                    f'Depth gate odom ENABLED — subscribing to: {self._df_gate_odom_topic}'
                )
        else:
            self.get_logger().info('Depth fusion DISABLED — no depth subscriber created')

        # Observer Self-Rejection (Layer 4) subscriber — zero overhead when disabled.
        # When enabled, this node subscribes to the other observer drone's main odometry
        # and uses it to project the drone's expected image-plane position, rejecting
        # any detection that falls inside the projected footprint.
        # Reference: Blackman & Popoli (1999), Chapter 7 — negative object recognition gate.
        if self.osr_enabled:
            # Determine which UAV name belongs to the other observer by inspecting
            # the configured role-to-uav mapping; fall back to a simple heuristic.
            _other_observer_uav: str = ''
            if _configured_uav_roles:
                # Primary path: explicit observer1/observer2 pair.
                if self.observer_role in ('observer1', 'observer2'):
                    _peer_role = 'observer2' if self.observer_role == 'observer1' else 'observer1'
                    _other_observer_uav = str(_configured_uav_roles.get(_peer_role, '') or '').strip()

                # Fallback path: pick the first observer role that is neither self role nor self UAV.
                if not _other_observer_uav:
                    for _role_name, _role_uav in _configured_uav_roles.items():
                        _role_name_norm = str(_role_name).strip().lower()
                        _role_uav_name = str(_role_uav or '').strip()
                        if not _role_uav_name:
                            continue
                        if _role_name_norm == str(self.observer_role).strip().lower():
                            continue
                        if _role_uav_name == self.uav_name:
                            continue
                        if 'observer' in _role_name_norm:
                            _other_observer_uav = _role_uav_name
                            break
            if _other_observer_uav:
                _osr_odom_topic = f'/{_other_observer_uav}/estimation_manager/odom_main'
                self.create_subscription(
                    Odometry, _osr_odom_topic, self._other_observer_odom_callback, 10
                )
                # [rev2026-07-14c] Explicit config echo so operators can grep the
                # startup log and confirm this process is actually running the
                # current source/config, not a stale pre-fix process. Includes
                # only_if_closer and gate_radius since those change gate behavior.
                self.get_logger().info(
                    f'Observer self-rejection ENABLED [rev2026-07-14e] — tracking {_other_observer_uav} '
                    f'on {_osr_odom_topic} | only_if_closer={self.osr_only_if_closer} '
                    f'gate_radius_m={self.osr_gate_radius_m} max_odom_age_sec={self.osr_max_odom_age_ns / 1e9:.2f}'
                )
            else:
                self.get_logger().warn(
                    'Observer self-rejection enabled but could not determine other-observer UAV name; '
                    'Layer 4 will have no effect.  Check uav_roles in runtime_stack.yaml.'
                )

        self.create_timer(1.0 / self.processing_rate_hz, self.processing_timer_callback)

        self.get_logger().info(f'YOLO model path: {self.model_path}')
        self.get_logger().info(f'Observer config role: {self.observer_role}')
        self.get_logger().info(f'Observer ({self.uav_name}): {self.image_topic} -> {self.detection_topic}')
        self.get_logger().info(f'Camera info topic: {self.camera_info_topic}')
        self.get_logger().info(f'Publishing frozen observations to: {self.observation_topic}')
        if self.publish_debug_annotated_image:
            self.get_logger().info(f'Publishing debug overlay to: /{self.uav_name}/debug_annotated/compressed')
        if self.publish_detector_evaluation_packets:
            self.get_logger().info(f'Publishing detector evaluation packets to: {self.detector_evaluation_topic}')
        if self.publish_camera_static_tf:
            self.get_logger().info('Camera static TF publishing enabled for bag/debug use')
        else:
            self.get_logger().info('Camera static TF publishing disabled; expect simulator/real stack to provide TF')
        self.get_logger().info(f'Processing rate: {self.processing_rate_hz:.1f} Hz')
        self.get_logger().info(
            f'Image-space KF: {"ENABLED" if self.kf_enabled else "DISABLED"}'
            f' (ego_motion={self._kf_tracker._ego_motion_enabled},'
            f' cam_aware={self._kf_tracker._cam_aware_gating_enabled},'
            f' n_confirm={self._kf_tracker._n_confirm},'
            f' max_age={self._kf_tracker._max_age},'
            f' ghost_timeout={self.kf_ghost_timeout_sec:.2f}s,'
            f' ghost_reentry={self.kf_ghost_reentry_m}/{self.kf_ghost_reentry_n},'
            f' ghost_spread={self.kf_ghost_reentry_spread_px:.1f}px)'
        )

    @staticmethod
    def _round_up_to_stride(value: float, stride: int = 32) -> int:
        value = max(float(stride), value)
        return int(math.ceil(value / float(stride)) * stride)

    @staticmethod
    def _resolve_observer_role(requested_role: str, matching_roles: List[str]) -> str:
        if requested_role in ('observer1', 'observer2'):
            return requested_role
        if len(matching_roles) == 1:
            return matching_roles[0]
        return 'observer2'

    def _resolve_camera_frame(self, camera_frame: str) -> str:
        if camera_frame and self.uav_name not in camera_frame:
            return f'{self.uav_name}/{camera_frame}'
        return camera_frame

    def _next_observation_id(self) -> int:
        """Return the next monotonically increasing observation id."""
        observation_id = self._observation_id_counter
        self._observation_id_counter = (self._observation_id_counter + 1) & 0xFFFFFFFF
        return observation_id

    def _lookup_measurement_transform(
        self,
        camera_frame: str,
        stamp_ns: int,
    ) -> Optional[Tuple[TransformStamped, int, int]]:
        """Resolve the transform used to freeze a single outgoing observation."""
        sec = stamp_ns // 1_000_000_000
        nanosec = stamp_ns % 1_000_000_000

        try:
            transform = self.tf_buffer.lookup_transform(
                self.world_frame_id,
                camera_frame,
                Time(seconds=sec, nanoseconds=nanosec),
                Duration(seconds=0),
            )
            return transform, 0, TF_STATUS_EXACT
        except Exception:
            if not self.allow_tf_fallback:
                self.get_logger().warn(
                    f'Exact TF unavailable for {camera_frame} at {stamp_ns}; dropping measurement.',
                    throttle_duration_sec=2.0,
                )
                return None

        try:
            transform = self.tf_buffer.lookup_transform(
                self.world_frame_id,
                camera_frame,
                Time(),
                Duration(seconds=0),
            )
        except Exception as exc:
            self.get_logger().warn(
                f'TF lookup {self.world_frame_id} <- {camera_frame} failed: {exc}',
                throttle_duration_sec=2.0,
            )
            return None

        transform_stamp_ns = stamp_to_ns(transform.header.stamp)
        if transform_stamp_ns <= 0:
            transform_stamp_ns = stamp_ns
        tf_delta_ns = transform_stamp_ns - stamp_ns
        if abs(tf_delta_ns) > self.max_tf_delta_ns:
            self.get_logger().warn(
                f'Dropping measurement because TF delta is {tf_delta_ns / 1e6:.1f} ms '
                f'(limit {self.max_tf_delta_ns / 1e6:.1f} ms).',
                throttle_duration_sec=2.0,
            )
            return None

        return transform, tf_delta_ns, TF_STATUS_FALLBACK_LATEST

    def _publish_detector_evaluation_packet(
        self,
        msg,
        decision_debug: Optional[Dict[str, object]],
        infer_ms: float,
        img_w: int,
        img_h: int,
        use_roi: bool,
        roi_x1: int,
        roi_y1: int,
        roi_x2: int,
        roi_y2: int,
        roi_pred_center: Optional[Tuple[float, float]],
        best_detection: Optional[DetectionObservation],
        kf_state_before_selection: str,
        kf_state_after_update: str,
        kf_update_accepted: Optional[bool],
        measurement_published: bool,
    ) -> None:
        if (
            not self.publish_detector_evaluation_packets
            or self.detector_evaluation_pub is None
            or decision_debug is None
        ):
            return

        camera_frame = ''
        if self.camera_info is not None:
            camera_frame = self._resolve_camera_frame(
                self._camera_frame_override or self.camera_info.header.frame_id
            )

        selected_detection = None
        if best_detection is not None:
            selected_detection = {
                'source_index': int(best_detection.source_index),
                'center_uv': [float(best_detection.center_u), float(best_detection.center_v)],
                'bbox_wh': [float(best_detection.bbox_w), float(best_detection.bbox_h)],
                'confidence': float(best_detection.confidence),
                'class_id': int(best_detection.class_id),
                'class_name': best_detection.class_name,
            }

        model_path_obj = FsPath(self.model_path)
        if self.model_path.endswith('.pt'):
            model_format = 'pt'
            model_weights_dir = str(model_path_obj.parent)
        elif model_path_obj.is_dir() or model_path_obj.name.endswith('_openvino_model') or 'openvino' in model_path_obj.name.lower():
            model_format = 'openvino'
            model_weights_dir = str(model_path_obj)
        else:
            model_format = model_path_obj.suffix.lstrip('.').lower() or 'unknown'
            model_weights_dir = str(model_path_obj.parent if model_path_obj.suffix else model_path_obj)

        payload = {
            'schema_version': 1,
            'frame_stamp_ns': int(stamp_to_ns(msg.header.stamp)),
            'frame_id': msg.header.frame_id,
            'camera_frame_id': camera_frame,
            'world_frame_id': self.world_frame_id,
            'uav_name': self.uav_name,
            'observer_role': self.observer_role,
            'model_path': self.model_path,
            'model_weights_dir': model_weights_dir,
            'model_format': model_format,
            'target_class_id': int(self.target_class_id),
            'confidence_threshold': float(self.confidence_threshold),
            'iou_threshold': float(self.iou_threshold),
            'imgsz': int(self.imgsz),
            'infer_ms': float(infer_ms),
            'image_width': int(img_w),
            'image_height': int(img_h),
            'mode': 'ROI' if use_roi else 'FULL',
            'roi': {
                'active': bool(use_roi),
                'bbox_xyxy': [int(roi_x1), int(roi_y1), int(roi_x2), int(roi_y2)] if use_roi else None,
                'pred_center_uv': [float(roi_pred_center[0]), float(roi_pred_center[1])] if roi_pred_center is not None else None,
                'consecutive_misses': int(self._roi_consecutive_misses),
            },
            'kf_enabled': bool(self.kf_enabled),
            'kf_state_before_selection': kf_state_before_selection,
            'kf_state_after_update': kf_state_after_update,
            'kf_confirmed_after_update': bool(self._kf_tracker.is_confirmed) if self.kf_enabled else False,
            'kf_update_accepted': kf_update_accepted,
            'kf_state_u_v_udot_vdot': self._kf_tracker._x.tolist() if self.kf_enabled and self._kf_tracker._x is not None else None,
            'kf_covariance_4x4': self._kf_tracker._P.tolist() if self.kf_enabled and self._kf_tracker._P is not None else None,
            'kf_last_nis': float(self._kf_tracker._nis_deque[-1]) if self.kf_enabled and len(self._kf_tracker._nis_deque) > 0 else None,
            'measurement_published': bool(measurement_published),
            'selected_detection': selected_detection,
            'detector': decision_debug,
        }

        packet_msg = String()
        packet_msg.data = json.dumps(payload, separators=(',', ':'), ensure_ascii=True)
        self.detector_evaluation_pub.publish(packet_msg)

    def _publish_camera_static_tf(self, config_path: FsPath) -> None:
        try:
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f)
            ct = cfg.get('camera_tf', {})
            role_tf = ct.get(self.observer_role, {})
            if not role_tf:
                raise KeyError(f'camera_tf.{self.observer_role}')
            child = ct.get('child_frame', 'rgbd_link')
            self._yaw_offset = float(role_tf.get('yaw_offset', 0.0))
        except Exception as exc:
            self._yaw_offset = 0.0
            self.get_logger().warn(
                f'No camera_tf config found for role {self.observer_role}, skipping static TF: {exc}'
            )
            return

        self._static_tf_broadcaster = StaticTransformBroadcaster(self)

        link_frame = f'{self.uav_name}/{child}'
        color_frame = f'{self.uav_name}/rgbd_color_frame'
        optical_frame = f'{self.uav_name}/rgbd_color_optical_frame'

        t1 = TransformStamped()
        t1.header.stamp = Time().to_msg()
        t1.header.frame_id = f'{self.uav_name}/fcu'
        t1.child_frame_id = link_frame
        t1.transform.translation.x = float(role_tf.get('x', 0.0))
        t1.transform.translation.y = float(role_tf.get('y', 0.0))
        t1.transform.translation.z = float(role_tf.get('z', 0.0))
        r = role_tf.get('roll', 0.0)
        p = role_tf.get('pitch', 0.0)
        y = role_tf.get('yaw', 0.0) + self._yaw_offset
        cy, sy = math.cos(y * 0.5), math.sin(y * 0.5)
        cp, sp = math.cos(p * 0.5), math.sin(p * 0.5)
        cr, sr = math.cos(r * 0.5), math.sin(r * 0.5)
        t1.transform.rotation.w = cr * cp * cy + sr * sp * sy
        t1.transform.rotation.x = sr * cp * cy - cr * sp * sy
        t1.transform.rotation.y = cr * sp * cy + sr * cp * sy
        t1.transform.rotation.z = cr * cp * sy - sr * sp * cy

        t2 = TransformStamped()
        t2.header.stamp = t1.header.stamp
        t2.header.frame_id = link_frame
        t2.child_frame_id = color_frame
        t2.transform.translation.x = -0.001
        t2.transform.translation.y = 0.015
        t2.transform.translation.z = -0.000
        t2.transform.rotation.x = -0.008
        t2.transform.rotation.y = 0.001
        t2.transform.rotation.z = 0.002
        t2.transform.rotation.w = 1.000

        t3 = TransformStamped()
        t3.header.stamp = t1.header.stamp
        t3.header.frame_id = color_frame
        t3.child_frame_id = optical_frame
        t3.transform.translation.x = 0.000
        t3.transform.translation.y = 0.000
        t3.transform.translation.z = 0.000
        t3.transform.rotation.x = -0.5
        t3.transform.rotation.y = 0.5
        t3.transform.rotation.z = -0.5
        t3.transform.rotation.w = 0.5

        self._static_tf_broadcaster.sendTransform([t1, t2, t3])
        self.get_logger().info(
            f'Static TF: {t1.header.frame_id} -> {t1.child_frame_id} '
            f'(xyz={t1.transform.translation.x:.3f},{t1.transform.translation.y:.3f},{t1.transform.translation.z:.3f})'
        )

    def camera_info_callback(self, msg: CameraInfo) -> None:
        self.camera_info = msg

    def _target_odom_callback(self, msg: Odometry) -> None:
        self._target_odom = msg
        self._target_odom_stamp_ns = stamp_to_ns(msg.header.stamp)

    def _tc_odom_callback(self, msg: Odometry) -> None:
        """Callback for the 3D-guided Tight Coupling feedback.

        [L2-bug1 fix] Also estimates linear acceleration via EMA-filtered backward
        finite difference of the twist.linear velocity.  This feeds the
        velocity-compensated projection in _get_tc_projection with a real CA
        extrapolation instead of the degenerate CV (a=0) fallback that was
        active before because _tc_latest_accel was never populated.

        Noise guard: if dt is < 5 ms (duplicate message) or > 500 ms (gap / restart),
        the finite difference is discarded and the previous estimate is kept.
        """
        stamp_ns = stamp_to_ns(msg.header.stamp)
        vel = msg.twist.twist.linear
        v_new = np.array([vel.x, vel.y, vel.z], dtype=np.float64)

        if self._tc_prev_vel is not None and self._tc_prev_vel_stamp_ns > 0:
            dt_ns = stamp_ns - self._tc_prev_vel_stamp_ns
            dt_sec = dt_ns / 1e9
            if 0.005 <= dt_sec <= 0.5:  # valid differentiation window
                a_raw = (v_new - self._tc_prev_vel) / dt_sec
                # EMA filter to suppress numerical differentiation noise
                self._tc_accel_estimate = (
                    self._tc_accel_ema_alpha * a_raw
                    + (1.0 - self._tc_accel_ema_alpha) * self._tc_accel_estimate
                )

        self._tc_prev_vel = v_new
        self._tc_prev_vel_stamp_ns = stamp_ns
        self._tc_latest_odom = msg
        self._tc_latest_odom_stamp_ns = stamp_ns

    def _df_gate_odom_callback(self, msg: Odometry) -> None:
        """Cache the latest IMM odometry for the depth consistency gate.

        [L3-bug2 fix] This is a dedicated subscriber created only when
        depth_fusion.enabled=true and gating_enabled=true, completely independent
        of tc_enabled.  Decoupling the two subscriptions ensures the depth gate
        is not silently disabled when the user sets tc_enabled=false.
        """
        self._df_gate_odom = msg
        self._df_gate_odom_stamp_ns = stamp_to_ns(msg.header.stamp)

    def _other_observer_odom_callback(self, msg: Odometry) -> None:
        """Cache the latest odometry of the other observer drone for Layer 4 OSR.

        [Layer 4] Observer Self-Rejection subscribes to the other observer's
        estimation_manager/odom_main and caches its pose for `_get_other_observer_gate`.
        """
        self._other_observer_odom = msg
        self._other_observer_odom_stamp_ns = stamp_to_ns(msg.header.stamp)

    def _get_other_observer_gate(
        self,
        stamp_ns: int,
        camera_matrix: np.ndarray,
    ) -> Optional[Tuple[float, float, float]]:
        """Project the sibling observer into this camera's image plane for OSR.

        [Layer 4] Observer Self-Rejection (OSR) — rejects false positives caused by one
        observer drone flying into the field-of-view of the other.

        Returns the projected pixel centre and pixel gate RADIUS (u_obs, v_obs,
        gate_px) for this frame, or None if the gate is inactive/unavailable
        (odom missing/stale, TF unresolved, degenerate geometry, or the
        foreground condition fails). This is computed ONCE per frame — the
        caller is responsible for selecting, among the detections that fall
        inside this disc, only the single nearest one as the actual sibling
        (Global Nearest Neighbor association within a validation gate — see
        Bar-Shalom, Li & Kirubarajan 2001, and Blackman & Popoli 1999, Ch. 7 —
        NOT a blanket "reject everything inside the disc" rule, which would
        incorrectly veto unrelated detections such as a distant real target
        that happens to share the same bearing neighborhood as a close sibling).

        Algorithm:
        1. Age-check the cached other-observer odometry.
        2. Velocity-propagate the other observer's world position to the detection timestamp.
        3. TF-transform the propagated position into the camera optical frame.
        4. Project to image coordinates via the pinhole model.
        5. Compute a pixel gate radius from the drone body radius:
               gate_px = (osr_gate_radius_m / Zc) * max(fx, fy)
        6. Optionally require the other observer to be closer than the IMM target estimate
           by at least `osr_depth_margin_m` (foreground condition).

        Cross-UAV frame note: the two observers do NOT share a connected tf2 tree
        (each UAV only publishes its own local TF subtree), so `odom.header.frame_id`
        (e.g. "uav2/gps_baro_origin") can never be resolved directly against this
        node's own camera frame. Following the same assumption already used
        elsewhere in this codebase for cross-UAV geometry (see
        triangulation_node._shares_common_world_origin), we treat the peer's
        origin-type frame as physically coincident (same origin & orientation)
        with the equivalently-named frame in THIS UAV's own tree — i.e.
        "uav2/<leaf>" ≡ "uav1/<leaf>" — and look up "uav1/<leaf> -> camera"
        entirely within our own, resolvable TF tree instead of attempting an
        impossible cross-tree lookup.

        References:
          Blackman & Popoli (1999) — Design and Analysis of Modern Tracking Systems,
          Chapter 7: negative object recognition gate.
          Hartley & Zisserman (2004) — Multiple View Geometry, §6.1: pinhole projection.

        Args:
            stamp_ns: Detection timestamp in nanoseconds.
            camera_matrix: 3×3 intrinsic matrix K = [[fx,0,cx],[0,fy,cy],[0,0,1]].

        Returns:
            (u_obs, v_obs, gate_px) if the sibling gate is active this frame, else None.
        """
        if not self.osr_enabled:
            return None
        odom = self._other_observer_odom
        if odom is None:
            return None
        # Symmetric skew window (same convention as the depth-fusion gate at
        # _check_depth_gate): stamp_ns is the IMAGE capture timestamp, while
        # _other_observer_odom_stamp_ns is set from the odometry message header
        # at cache time. Odometry typically publishes far faster than the
        # camera/YOLO pipeline, so by the time this runs a newer odom sample
        # than the image's own capture stamp is routinely already cached —
        # that is NOT staleness, just normal pipeline latency, and must not
        # zero out the gate. Keep the signed delta for velocity extrapolation
        # (dt can legitimately be negative: back-extrapolate a too-new sample).
        delta_ns = stamp_ns - self._other_observer_odom_stamp_ns
        if abs(delta_ns) > self.osr_max_odom_age_ns:
            return None

        # 1. Extract other-observer world position and velocity
        pos = odom.pose.pose.position
        vel = odom.twist.twist.linear
        dt = float(delta_ns) * 1e-9

        p_world = np.array([pos.x + vel.x * dt,
                             pos.y + vel.y * dt,
                             pos.z + vel.z * dt], dtype=np.float64)

        # 2. TF: peer's origin frame (assumed coincident with our own
        # equivalently-named frame) → camera frame, time-matched to the
        # detection's own timestamp (stamp_ns) — NOT "latest" — because this
        # node (the camera) is itself free-flying and can rotate/translate
        # between the image's capture instant and the (slightly later) instant
        # this callback runs. Mirrors the exact-then-fallback-with-delta-check
        # pattern used everywhere else in this file (see
        # _lookup_measurement_transform).
        _cam_frame = self._resolve_camera_frame(
            self._camera_frame_override
            or (self.camera_info.header.frame_id if self.camera_info else '')
        )
        if not _cam_frame:
            return None
        _peer_origin_leaf = odom.header.frame_id.rsplit('/', 1)[-1] if odom.header.frame_id else ''
        if not _peer_origin_leaf:
            return None
        _local_equivalent_frame = f'{self.uav_name}/{_peer_origin_leaf}'
        _sec = stamp_ns // 1_000_000_000
        _nanosec = stamp_ns % 1_000_000_000
        try:
            tf_stamped = self.tf_buffer.lookup_transform(
                _cam_frame,
                _local_equivalent_frame,
                rclpy.time.Time(seconds=_sec, nanoseconds=_nanosec),
                timeout=rclpy.duration.Duration(seconds=self.tf_timeout_sec),
            )
        except Exception:
            # Exact-time TF unavailable (buffer too short / no extrapolation) —
            # fall back to latest, but only accept it if it is not stale
            # relative to the detection timestamp (same budget as
            # vision.max_tf_delta_ms used for outgoing measurements).
            try:
                tf_stamped = self.tf_buffer.lookup_transform(
                    _cam_frame,
                    _local_equivalent_frame,
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=self.tf_timeout_sec),
                )
            except Exception:
                self.get_logger().warn(
                    f'OSR: TF lookup {_cam_frame} <- {_local_equivalent_frame} failed. This local-tree '
                    f'frame should always resolve; check that {_local_equivalent_frame} actually exists '
                    'in this UAV\'s own TF tree.',
                    throttle_duration_sec=5.0,
                )
                return None
            _tf_stamp_ns = stamp_to_ns(tf_stamped.header.stamp)
            if _tf_stamp_ns > 0 and abs(_tf_stamp_ns - stamp_ns) > self.max_tf_delta_ns:
                return None

        t = tf_stamped.transform.translation
        q = tf_stamped.transform.rotation
        T_cw = np.array([t.x, t.y, t.z], dtype=np.float64)
        R_cw = quat_to_rotmat(q.x, q.y, q.z, q.w)

        p_cam = R_cw @ p_world + T_cw
        Zc = float(p_cam[2])
        if Zc <= 0.1:
            return None

        # 3. Project to image plane
        fx = float(camera_matrix[0, 0])
        fy = float(camera_matrix[1, 1])
        cx = float(camera_matrix[0, 2])
        cy = float(camera_matrix[1, 2])
        u_obs = fx * p_cam[0] / Zc + cx
        v_obs = fy * p_cam[1] / Zc + cy

        # 4. Pixel gate radius. NOTE: this scales as 1/Zc — at close range the
        # sibling's physically-motivated body radius legitimately projects to a
        # LARGE pixel disc (it genuinely fills much of the frame), which can
        # spuriously encompass an unrelated, distant detection (e.g. the real
        # target far in the background) that merely shares a similar bearing.
        # That is exactly why the caller must pick only the single nearest
        # in-gate detection rather than rejecting every detection inside this
        # disc — see the docstring above.
        gate_px = (self.osr_gate_radius_m / Zc) * max(fx, fy)

        # 5. Foreground depth condition (optional)
        if self.osr_only_if_closer:
            # Compare Zc of other observer vs IMM target estimate.
            # _tc_latest_odom and _df_gate_odom are only populated when
            # tight_coupling.enabled / depth_fusion.enabled(+gating) are true;
            # fall back to _target_odom (unconditionally subscribed to the same
            # /target/odometry_estimate topic) so the foreground check still
            # works with the loose-coupling-only default configuration.
            target_zc: Optional[float] = None
            target_odom = (
                getattr(self, '_tc_latest_odom', None)
                or self._df_gate_odom
                or self._target_odom
            )
            if target_odom is not None:
                t_pos = target_odom.pose.pose.position
                p_tgt_world = np.array([t_pos.x, t_pos.y, t_pos.z], dtype=np.float64)
                p_tgt_cam = R_cw @ p_tgt_world + T_cw
                target_zc = float(p_tgt_cam[2])
            if target_zc is None or (Zc >= target_zc - self.osr_depth_margin_m):
                return None

        self._last_other_observer_proj = (u_obs, v_obs)  # for draw_predictions debug marker
        return (u_obs, v_obs, gate_px)

    def _depth_callback(self, msg: Image) -> None:
        """Cache the latest aligned depth frame for extraction in process_image.

        Encoding handling (Hartley & Zisserman, Ch. 6 — depth along ray):
          - 16UC1: integer millimetres (Intel RealSense / Azure Kinect default)
          - 32FC1: float metres (simulation / OpenNI-style streams)
        The conversion to metres is deferred to _extract_depth_from_bbox so
        that the raw image bytes can be stored without any allocation beyond
        the numpy view.
        """
        if not self.df_enabled:
            return
        stamp_ns = stamp_to_ns(msg.header.stamp)
        try:
            if msg.encoding == '16UC1':
                depth_arr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough').astype(np.float32) * 1e-3
            elif msg.encoding == '32FC1':
                depth_arr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough').astype(np.float32)
            else:
                # Unknown encoding — attempt passthrough; values remain raw
                depth_arr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough').astype(np.float32)
                self.get_logger().warn(
                    f'Depth encoding {msg.encoding!r} not explicitly handled; treating as metres.',
                    throttle_duration_sec=10.0,
                )
        except Exception as exc:
            self.get_logger().warn(
                f'Depth decode failed: {exc}', throttle_duration_sec=2.0
            )
            return
        self._depth_cache.append((stamp_ns, depth_arr))

    def _lookup_depth_at_stamp(self, stamp_ns: int) -> Optional[np.ndarray]:
        """Nearest-neighbour lookup of the cached depth frame closest to stamp_ns.

        Returns None if the cache is empty or the closest frame exceeds df_max_align_ns.
        """
        if not self._depth_cache:
            return None
        best_diff = self.df_max_align_ns + 1
        best_arr: Optional[np.ndarray] = None
        for cached_ns, cached_arr in self._depth_cache:
            diff = abs(cached_ns - stamp_ns)
            if diff < best_diff:
                best_diff = diff
                best_arr = cached_arr
        if best_diff > self.df_max_align_ns:
            return None
        return best_arr

    def _extract_depth_from_bbox(
        self,
        depth_arr: np.ndarray,
        u1: float,
        v1: float,
        u2: float,
        v2: float,
    ) -> Tuple[bool, float, float]:
        """Sample the depth inside a YOLO bounding box and return (valid, z_m, sigma_z_m).

        Robust aggregation strategy (Khoshelham & Elberink 2012, Trevor et al.):
          - 'median'       : robust to holes and flying pixels (default)
          - 'center_pixel' : fast, uses only the bbox centroid
          - 'trimmed_mean' : discards top/bottom df_trimmed_frac before averaging

        Noise model (Khoshelham & Elberink 2012, Eq. 5):
          sigma_z = sigma_z_base + sigma_z_range_coeff * z_meas
        """
        h, w = depth_arr.shape[:2]
        # Clip bbox to image bounds
        x1 = int(max(0, min(w - 1, math.floor(u1))))
        y1 = int(max(0, min(h - 1, math.floor(v1))))
        x2 = int(max(x1 + 1, min(w, math.ceil(u2))))
        y2 = int(max(y1 + 1, min(h, math.ceil(v2))))

        if self.df_sample_method == 'center_pixel':
            cu = int(round(0.5 * (u1 + u2)))
            cv = int(round(0.5 * (v1 + v2)))
            cu = max(0, min(w - 1, cu))
            cv = max(0, min(h - 1, cv))
            z = float(depth_arr[cv, cu])
            if not math.isfinite(z) or z < self.df_min_range or z > self.df_max_range:
                return False, 0.0, 0.0
            sigma = self.df_sigma_base + self.df_sigma_coeff * z
            return True, z, sigma

        # Extract patch with optional stride subsampling (CPU savings for large bboxes)
        stride = self.df_stride
        bbox_area = (x2 - x1) * (y2 - y1)
        if bbox_area > 4000 and stride < 2:
            stride = 2
        patch = depth_arr[y1:y2:stride, x1:x2:stride]
        flat = patch.flatten()

        # Keep only finite values within the valid sensor range
        valid_mask = np.isfinite(flat) & (flat >= self.df_min_range) & (flat <= self.df_max_range)
        valid_count = int(np.sum(valid_mask))
        total_count = max(1, len(flat))
        if (valid_count / total_count) < self.df_min_valid_frac:
            return False, 0.0, 0.0

        valid_vals = flat[valid_mask]

        if self.df_sample_method == 'trimmed_mean':
            n = len(valid_vals)
            k = max(0, int(math.floor(self.df_trimmed_frac * n)))
            sorted_vals = np.sort(valid_vals)
            trimmed = sorted_vals[k: n - k] if n - 2 * k > 0 else sorted_vals
            z = float(np.mean(trimmed))
        else:
            # default: median — most robust for flying pixels and holes
            z = float(np.median(valid_vals))

        if not math.isfinite(z) or z < self.df_min_range or z > self.df_max_range:
            return False, 0.0, 0.0

        sigma = self.df_sigma_base + self.df_sigma_coeff * z
        return True, z, sigma

    def _check_depth_gate(
        self,
        z_meas: float,
        sigma_z_meas: float,
        stamp_ns: int,
        camera_frame: str,
        detection_ray_optical: Optional[np.ndarray] = None,
    ) -> bool:
        """Scalar depth consistency gate against the 3D IMM odometry estimate.

        Rejects a near occluder (collinear in bearing, different range) when:
            |r_meas - r_pred| > gate_k * sqrt(sigma_r_meas^2 + sigma_r_pred^2)

        where r_pred is the range along the DETECTION RAY (not Z_camera).

        [L3-bug1 fix] Previously used z_pred = p_cam[2] (Z-axis of camera frame),
        which equals the true range only for on-axis detections.  For targets at
        the edge of the FOV the Z-component underestimates range, producing a
        systematic bias that tightens the gate for off-axis targets.

        Corrected formulation: project p_cam onto the unit detection ray d_hat.
        This gives the true along-ray distance regardless of off-axis angle:
            r_pred = d_hat · p_cam,  where d_hat is normalised in camera frame.
        The Jacobian for covariance propagation becomes:
            ∂r/∂p_world = d_hat_cam^T @ R_cw
        References:
          Hartley & Zisserman (2004) Ch.6 — projection geometry.
          Bar-Shalom, Li & Kirubarajan (2001) Sec. 3.4 — scalar 1-D gate.

        [L3-bug2 fix] The gate now uses self._df_gate_odom (a dedicated IMM odom
        subscriber created whenever depth_fusion.enabled=true and gating_enabled=true),
        independent of tc_enabled.  Previously the gate silently bypassed when TC was
        disabled because it reused _tc_latest_odom which was only subscribed when
        tc_enabled=true.

        Returns True if the detection PASSES the gate (i.e. consistent with
        the 3D estimate), False if it should be rejected as an occluder.
        Gate is bypassed (returns True) when:
          - gating_enabled=false
          - df_enabled=false  (redundant guard)
          - odometry too old or unavailable
          - tr(P_pos) > max_odom_covariance_trace (filter coasting, gate unreliable)
          - detection_ray_optical is None (fallback to Z-axis approximation)
        """
        if not self.df_gating or not self.df_enabled:
            return True  # gate disabled — always pass

        # [L3-bug2 fix] Use the dedicated depth-gate odom, not _tc_latest_odom.
        odom = self._df_gate_odom
        odom_stamp_ns = self._df_gate_odom_stamp_ns
        if odom is None:
            return True
        age_ns = abs(stamp_ns - odom_stamp_ns)
        if age_ns > self.df_max_odom_age_ns:
            return True

        cov6 = np.array(odom.pose.covariance, dtype=np.float64).reshape(6, 6)
        P_pos = cov6[:3, :3]
        if float(np.trace(P_pos)) > self.df_max_odom_cov:
            return True

        # Project target world position into camera frame
        try:
            stamp_sec = stamp_ns // 1_000_000_000
            stamp_nsec = stamp_ns % 1_000_000_000
            T = self.tf_buffer.lookup_transform(
                camera_frame,
                odom.header.frame_id,
                Time(seconds=stamp_sec, nanoseconds=stamp_nsec),
                Duration(seconds=0),
            )
        except Exception:
            try:
                T = self.tf_buffer.lookup_transform(
                    camera_frame, odom.header.frame_id, Time(), Duration(seconds=0)
                )
            except Exception:
                return True  # no TF — do not veto

        t_v = T.transform.translation
        q_v = T.transform.rotation
        R_cw = quat_to_rotmat(q_v.x, q_v.y, q_v.z, q_v.w)
        t_cw = np.array([t_v.x, t_v.y, t_v.z], dtype=np.float64)
        pos = odom.pose.pose.position
        p_cam = R_cw @ np.array([pos.x, pos.y, pos.z], dtype=np.float64) + t_cw

        # [L3-bug1 fix] Project p_cam onto the detection ray for geometrically
        # correct range prediction at any off-axis angle.
        if detection_ray_optical is not None:
            # Normalize the detection ray in camera frame
            d_norm = float(np.linalg.norm(detection_ray_optical))
            if d_norm > 1e-6:
                d_hat = detection_ray_optical / d_norm
                r_pred = float(np.dot(d_hat, p_cam))
                if r_pred <= 0.0:
                    return True  # target behind camera along detection ray
                # Jacobian for range along ray: ∂r/∂p_world = d_hat^T @ R_cw
                dr_dp = d_hat @ R_cw  # shape (3,)
            else:
                # Degenerate ray — fall back to Z-axis approximation
                r_pred = float(p_cam[2])
                dr_dp = R_cw[2, :]
        else:
            # No ray provided — fall back to Z-axis approximation (original behavior)
            r_pred = float(p_cam[2])
            dr_dp = R_cw[2, :]

        if r_pred <= 0.0:
            return True  # target behind camera — gate unreliable

        # Propagate positional covariance to sigma_r_pred via linearised Jacobian
        sigma_r_pred = float(np.sqrt(max(1e-9, dr_dp @ P_pos @ dr_dp)))

        combined_sigma = math.sqrt(sigma_z_meas ** 2 + sigma_r_pred ** 2)
        return abs(z_meas - r_pred) <= self.df_gate_k * combined_sigma

    def _get_roi_center(self, stamp_ns: int, img_w: int, img_h: int) -> Optional[Tuple[float, float]]:
        """Predict the ROI center from the image-plane track first, then fall back to target odometry."""
        
        # 1. Primary: Use visual track (if recently updated and not missing)
        if self._roi_last_center is not None and self._roi_last_stamp_ns > 0 and self._roi_consecutive_misses == 0:
            dt = (stamp_ns - self._roi_last_stamp_ns) / 1_000_000_000.0
            pred_u = self._roi_last_center[0] + self._roi_vel_u * dt
            pred_v = self._roi_last_center[1] + self._roi_vel_v * dt
            return (float(pred_u), float(pred_v))
            
        # 2. Fallback: Use 3D Target Odometry projection if visual track is lost/missing
        if self._target_odom is not None and self.camera_info is not None:
            odom_age_ns = abs(stamp_ns - self._target_odom_stamp_ns)
            if odom_age_ns < 2_000_000_000:
                camera_frame = self._resolve_camera_frame(
                    self._camera_frame_override or self.camera_info.header.frame_id
                )
                try:
                    lookup_time = Time(nanoseconds=stamp_ns)
                    transform = self.tf_buffer.lookup_transform(
                        camera_frame, self._target_odom.header.frame_id, lookup_time, Duration(seconds=0)
                    )
                    t = transform.transform.translation
                    q = transform.transform.rotation
                    r_cw = quat_to_rotmat(q.x, q.y, q.z, q.w)
                    t_cw = np.array([t.x, t.y, t.z], dtype=np.float64)

                    pos_3d = np.array(
                        [
                            self._target_odom.pose.pose.position.x,
                            self._target_odom.pose.pose.position.y,
                            self._target_odom.pose.pose.position.z,
                        ],
                        dtype=np.float64,
                    )
                    vel_3d = np.array(
                        [
                            self._target_odom.twist.twist.linear.x,
                            self._target_odom.twist.twist.linear.y,
                            self._target_odom.twist.twist.linear.z,
                        ],
                        dtype=np.float64,
                    )
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

        if self._roi_last_center is not None and (stamp_ns - self._roi_last_stamp_ns) < self.roi_timeout_ns:
            dt_sec = (stamp_ns - self._roi_last_stamp_ns) / 1_000_000_000.0
            pred_u = self._roi_last_center[0] + self._roi_vel_u * dt_sec
            pred_v = self._roi_last_center[1] + self._roi_vel_v * dt_sec
            return (pred_u, pred_v)

        return None

    def _clear_ghost_reentry_state(self) -> None:
        self._ghost_reentry_buffer.clear()
        self._ghost_reentry_ready = False

    def _ghost_reentry_anchor_radius_px(self, roi_half_side: float) -> float:
        reentry_spread_px = self.kf_ghost_reentry_spread_px if self.kf_ghost_reentry_spread_px > 0.0 else self.roi_init_spread
        return max(48.0, 2.0 * reentry_spread_px, min(0.25 * float(roi_half_side), 320.0))

    def _check_ghost_reentry_gate(
        self,
        center_u: float,
        center_v: float,
        confidence: float,
        stamp_ns: int,
        roi_pred_center: Optional[Tuple[float, float]],
        roi_half_side: float,
    ) -> bool:
        self._ghost_reentry_ready = False
        reentry_spread_px = self.kf_ghost_reentry_spread_px if self.kf_ghost_reentry_spread_px > 0.0 else self.roi_init_spread

        while self._ghost_reentry_buffer and (stamp_ns - self._ghost_reentry_buffer[0].stamp_ns) > self._kf_tracker._ghost_timeout_ns:
            self._ghost_reentry_buffer.popleft()

        if roi_pred_center is None or confidence < self.kf_ghost_reentry_conf_min:
            return False

        pred_u, pred_v = roi_pred_center
        anchor_radius_px = self._ghost_reentry_anchor_radius_px(roi_half_side)
        pred_dist_sq = (center_u - pred_u) ** 2 + (center_v - pred_v) ** 2
        if pred_dist_sq > (anchor_radius_px ** 2):
            return False

        self._ghost_reentry_buffer.append(
            GhostReentryObservation(
                stamp_ns=stamp_ns,
                center_u=center_u,
                center_v=center_v,
                confidence=confidence,
            )
        )

        if len(self._ghost_reentry_buffer) < self.kf_ghost_reentry_m:
            return False

        observations = list(self._ghost_reentry_buffer)
        for ref_obs in observations:
            close = sum(
                1
                for obs in observations
                if (
                    abs(obs.center_u - ref_obs.center_u) < reentry_spread_px
                    and abs(obs.center_v - ref_obs.center_v) < reentry_spread_px
                    and obs.confidence >= self.kf_ghost_reentry_conf_min
                )
            )
            if close >= self.kf_ghost_reentry_m:
                self._ghost_reentry_ready = True
                return True

        return False

    def _check_init_gate(self, center_u: float, center_v: float, stamp_ns: int) -> bool:
        """Arm ROI tracking only after enough recent detections cluster in image space.

        [L1 fix] Stale entries are pruned by timestamp before the M-of-N check.
        Previously the deque was bounded only by maxlen, meaning detections from
        a previous track episode (before KF reset) could persist and artificially
        satisfy the M-of-N condition during re-acquisition, causing premature
        initialisation on a false positive.  The pruning window matches the KF
        ghost timeout so that no entry older than that can contribute.
        """
        self._init_gate_buffer.append((center_u, center_v, stamp_ns))
        # Prune entries older than ghost_timeout_ns to prevent cross-episode contamination
        cutoff_ns = stamp_ns - self._kf_tracker._ghost_timeout_ns
        while self._init_gate_buffer and self._init_gate_buffer[0][2] < cutoff_ns:
            self._init_gate_buffer.popleft()
        if len(self._init_gate_buffer) < self.roi_init_m:
            return False
        centers = [(u, v) for u, v, _ in self._init_gate_buffer]
        for index in range(len(centers)):
            ref_u, ref_v = centers[index]
            close = sum(
                1
                for u, v in centers
                if abs(u - ref_u) < self.roi_init_spread and abs(v - ref_v) < self.roi_init_spread
            )
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
        """Run YOLO on the full image or ROI crop, then publish the selected observation and debug outputs."""
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

        use_roi = False
        roi_x1 = roi_y1 = roi_x2 = roi_y2 = 0
        infer_image = bgr_image
        infer_imgsz = self.imgsz
        roi_debug_details = ''

        if self.roi_crop_enabled and self._roi_last_bbox is not None:
            force_full = self._roi_frame_counter % self.roi_full_scan_interval == 0

            roi_center = self._get_roi_center(stamp_ns, img_w, img_h)
            has_track = roi_center is not None

            if has_track and not force_full:
                pred_u, pred_v = roi_center
                scale_preserving_w = int(self.roi_reference_imgsz * img_w / self.imgsz)
                scale_preserving_h = int(self.roi_reference_imgsz * img_h / self.imgsz)

                roi_w = int(scale_preserving_w * self.roi_margin)
                roi_h = int(scale_preserving_h * self.roi_margin)

                if self._roi_last_bbox is not None:
                    bx1, by1, bx2, by2 = self._roi_last_bbox
                    last_bbox_w = bx2 - bx1
                    last_bbox_h = by2 - by1
                    roi_w = max(roi_w, int(last_bbox_w * 2.5))
                    roi_h = max(roi_h, int(last_bbox_h * 2.5))

                if self._roi_consecutive_misses >= 3:
                    roi_w = int(roi_w * 2.0)
                    roi_h = int(roi_h * 2.0)
                elif self._roi_consecutive_misses >= 1:
                    roi_w = int(roi_w * 1.5)
                    roi_h = int(roi_h * 1.5)

                roi_x1 = int(max(0.0, pred_u - roi_w * 0.5))
                roi_y1 = int(max(0.0, pred_v - roi_h * 0.5))
                roi_x2 = int(min(float(img_w), roi_x1 + roi_w))
                roi_y2 = int(min(float(img_h), roi_y1 + roi_h))

                if (roi_x2 - roi_x1) >= self.roi_min_size and (roi_y2 - roi_y1) >= self.roi_min_size:
                    crop_w = roi_x2 - roi_x1
                    crop_h = roi_y2 - roi_y1
                    full_max_dim = float(max(img_w, img_h))
                    crop_max_dim = float(max(crop_w, crop_h))
                    infer_imgsz = min(
                        self.imgsz,
                        self._round_up_to_stride(self.imgsz * (crop_max_dim / full_max_dim)),
                    )
                    infer_image = bgr_image[roi_y1:roi_y2, roi_x1:roi_x2]
                    use_roi = True
                    roi_debug_details = (
                        f', roi={crop_w}x{crop_h}->{infer_imgsz}'
                        f', base={scale_preserving_w}x{scale_preserving_h}'
                        f', ref={self.roi_reference_imgsz}'
                    )

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
        self._record_log_window_frame(infer_ms, use_roi)

        roi_pred_center = None
        if use_roi and self._roi_last_center is not None:
            roi_pred_center = self._roi_last_center
        # When KF is actively tracking, override the selection center with the KF
        # predicted centroid. This ensures the candidate closest to the tracked target
        # is chosen even when a false positive has higher YOLO confidence.
        # Critical case: drone conf=0.3 vs tree conf=0.4 — geometric consistency wins.
        if (
            self.kf_enabled
            and self._kf_tracker.predicted_center is not None
            and self._kf_tracker.state != ImageSpaceKalmanTracker.UNINITIALIZED
        ):
            roi_pred_center = self._kf_tracker.predicted_center

        camera_matrix = None
        if self.camera_info is not None:
            camera_matrix = np.array(self.camera_info.k).reshape(3, 3)

        detector_eval_debug = {} if (self.publish_detector_evaluation_packets or self.publish_annotated_image) else None
        kf_state_before_selection = self._kf_tracker.state if self.kf_enabled else 'DISABLED'

        # Pre-compute TC projection for both the gate and the debug overlay.
        # This call is O(1) — one matrix multiply — and returns None when TC is
        # disabled/stale (see _get_tc_projection docstring for safety conditions).
        tc_projection: Optional[Tuple[np.ndarray, np.ndarray]] = None
        if self.tc_enabled and camera_matrix is not None:
            tc_projection = self._get_tc_projection(stamp_ns, camera_matrix)

        # Depth frame lookup — nearest-neighbour within df_max_align_ns.
        # Returns None immediately when df_enabled=False (no cache entries).
        depth_arr: Optional[np.ndarray] = None
        if self.df_enabled:
            depth_arr = self._lookup_depth_at_stamp(stamp_ns)

        best_detection = self.extract_best_detection(
            results,
            msg,
            roi_pred_center,
            roi_x1,
            roi_y1,
            roi_x2 - roi_x1 if use_roi else img_w,
            camera_matrix=camera_matrix,
            debug_sink=detector_eval_debug,
            tc_projection=tc_projection,
            depth_arr=depth_arr,
        )

        if best_detection is not None and use_roi:
            best_detection.center_u += roi_x1
            best_detection.center_v += roi_y1

        # ── KF-disabled path: original behavior ─────────────────────
        if not self.kf_enabled:
            if best_detection is not None:
                if self._roi_last_bbox is not None:
                    self._update_roi_tracker(best_detection, results, stamp_ns, roi_x1, roi_y1)
                    self._roi_consecutive_misses = 0
                else:
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
                annotated_image = self.draw_predictions(
                    bgr_image, results, best_detection, use_roi,
                    roi_x1, roi_y1, roi_x2, roi_y2,
                    kf_tracker=None,
                    debug_sink=detector_eval_debug,
                )
                comp_msg = CompressedImage()
                comp_msg.header = msg.header
                comp_msg.format = 'jpeg'
                comp_msg.data = bytes(cv2.imencode('.jpg', annotated_image, [cv2.IMWRITE_JPEG_QUALITY, 75])[1])
                self.detection_pub.publish(comp_msg)
                self._log_annotated_publish_count += 1

            if best_detection is not None:
                local_ray_data = self.compute_and_publish_ray(best_detection, depth_arr=depth_arr)
            else:
                local_ray_data = None

            self._publish_detector_evaluation_packet(
                msg,
                detector_eval_debug,
                infer_ms,
                img_w,
                img_h,
                use_roi,
                roi_x1,
                roi_y1,
                roi_x2,
                roi_y2,
                roi_pred_center,
                best_detection,
                kf_state_before_selection,
                'DISABLED',
                None,
                local_ray_data is not None,
            )

            if self.publish_debug_annotated_image and self.camera_info is not None:
                self.publish_ray_debug_image(bgr_image, self.camera_info, local_ray_data, msg.header)

            self.frame_counter += 1
            now = time.time()
            if now - self.last_log_time >= 2.0:
                fps = self.frame_counter / (now - self.last_log_time)
                avg_infer_ms = self._log_infer_total_ms / max(1, self.frame_counter)
                mode_summary = self._format_log_window_modes()
                esc = f' esc={self._roi_consecutive_misses}' if self._roi_consecutive_misses > 0 else ''
                self.get_logger().info(
                    f'Observer node: FPS={fps:.2f}, inference_avg={avg_infer_ms:.1f}ms, '
                    f'inference_max={self._log_infer_max_ms:.1f}ms, {mode_summary}{esc}, '
                    f'dropped={self._dropped_frames}, annotated_pub={self._log_annotated_publish_count}'
                    f'{roi_debug_details}'
                )
                self._reset_log_window(now)
            return

        # ── KF-enabled path ──────────────────────────────────────────

        # Compute the TF-based predicted pixel for KF mean override
        kf_roi_pred_center = self._get_roi_center(stamp_ns, img_w, img_h)

        # Compute R_ego: rotation from previous to current camera frame.
        # This is used by the tracker for ego-motion compensation (TAKF Mechanism 1).
        R_ego = None
        if self._kf_tracker_prev_stamp_ns > 0 and self.camera_info is not None:
            try:
                camera_frame = self._resolve_camera_frame(
                    self._camera_frame_override or self.camera_info.header.frame_id
                )
                T_curr = self.tf_buffer.lookup_transform(
                    self.world_frame_id, camera_frame,
                    Time(nanoseconds=stamp_ns), Duration(seconds=0)
                )
                T_prev = self.tf_buffer.lookup_transform(
                    self.world_frame_id, camera_frame,
                    Time(nanoseconds=self._kf_tracker_prev_stamp_ns),
                    Duration(seconds=0)
                )
                q = T_curr.transform.rotation
                R_wc_curr = quat_to_rotmat(q.x, q.y, q.z, q.w)
                q = T_prev.transform.rotation
                R_wc_prev = quat_to_rotmat(q.x, q.y, q.z, q.w)
                # R_ego maps a ray from camera frame at t_{k-1} to camera frame at t_k
                R_ego = R_wc_curr.T @ R_wc_prev
            except Exception:
                pass
        self._kf_tracker_prev_stamp_ns = stamp_ns

        # Camera intrinsic matrix is already built above for pre-filtering

        # Initialize the KF when the re-entry gate or the standard M-of-N gate fires
        if best_detection is not None and self._kf_tracker.state == ImageSpaceKalmanTracker.UNINITIALIZED:
            init_via_reentry = self._ghost_reentry_ready
            if init_via_reentry or self._check_init_gate(best_detection.center_u, best_detection.center_v, stamp_ns):
                self._kf_tracker.initialize(
                    best_detection.center_u, best_detection.center_v, stamp_ns,
                    best_detection.bbox_w, best_detection.bbox_h
                )
                self._kf_tracker.clear_ghost()
                self._init_gate_buffer.clear()
                self._clear_ghost_reentry_state()
                if init_via_reentry:
                    self.get_logger().info('[KF] Track initialized via ghost re-entry gate → TENTATIVE')
                else:
                    self.get_logger().info('[KF] Track initialized via M-of-N gate → TENTATIVE')

        # Run one predict + gate + update cycle on the tracker
        accepted, x_filtered = self._kf_tracker.update(
            detection=best_detection,
            stamp_ns=stamp_ns,
            camera_matrix=camera_matrix,
            R_ego=R_ego,
            roi_pred_center=kf_roi_pred_center,
            image_size=(img_w, img_h),
        )

        if self._kf_tracker.last_prediction_guard_reason is not None:
            self.get_logger().warn(
                f'[KF] Prediction guard engaged: {self._kf_tracker.last_prediction_guard_reason}',
                throttle_duration_sec=1.0,
            )

        # Publish ray only when track is CONFIRMED and detection passed the gate
        local_ray_data = None
        if self._kf_tracker.is_confirmed and x_filtered is not None:
            filtered_det = dataclasses.replace(
                best_detection,
                center_u=float(x_filtered[0]),
                center_v=float(x_filtered[1]),
            )
            local_ray_data = self.compute_and_publish_ray(filtered_det, depth_arr=depth_arr)

        # NIS consistency warning: possible persistent false positive
        if self._kf_tracker.is_confirmed and not self._kf_tracker.is_nis_consistent:
            mean_nis = float(np.mean(self._kf_tracker._nis_deque))
            self.get_logger().warn(
                f'[KF] NIS inconsistency: mean={mean_nis:.2f} > {self._kf_tracker._nis_threshold} — possible FP',
                throttle_duration_sec=3.0,
            )

        # Handle LOST state: full reset of KF, init gate, and prev stamp
        if self._kf_tracker.state == ImageSpaceKalmanTracker.LOST:
            self.get_logger().info(
                f'[KF] Track LOST after {self._kf_tracker._max_age} consecutive misses'
            )
            self._kf_tracker.reset()
            self._init_gate_buffer.clear()
            self._clear_ghost_reentry_state()
            self._kf_tracker_prev_stamp_ns = 0

        # Log state transitions
        if self._kf_tracker.is_confirmed and accepted and self._kf_tracker._consecutive_hits == self._kf_tracker._n_confirm:
            self.get_logger().info(
                f'[KF] Track CONFIRMED after {self._kf_tracker._n_confirm} consecutive hits'
            )

        # Update ROI tracking state only when KF has accepted and confirmed the detection.
        # This prevents false positives from corrupting the ROI crop prediction.
        if accepted and self._kf_tracker.is_confirmed and best_detection is not None:
            if self._roi_last_bbox is not None:
                self._update_roi_tracker(best_detection, results, stamp_ns, roi_x1, roi_y1)
                self._roi_consecutive_misses = 0
            else:
                self._update_roi_tracker(best_detection, results, stamp_ns, roi_x1, roi_y1)
                self._roi_consecutive_misses = 0
                self.get_logger().info('ROI track initialized via confirmed KF track')
        elif self.roi_crop_enabled and self._roi_last_bbox is not None:
            if best_detection is None or not accepted:
                self._roi_consecutive_misses += 1
            if (stamp_ns - self._roi_last_stamp_ns) >= self.roi_timeout_ns:
                self._roi_last_bbox = None
                self._roi_last_center = None
                self._roi_consecutive_misses = 0

        # Publish annotated image with detection overlay
        if self.publish_annotated_image:
            annotated_image = self.draw_predictions(
                bgr_image, results, best_detection, use_roi,
                roi_x1, roi_y1, roi_x2, roi_y2,
                kf_tracker=self._kf_tracker,
                camera_matrix=camera_matrix,
                debug_sink=detector_eval_debug,
                tc_projection=tc_projection,
            )
            # Draw KF debug overlay: predicted center crosshair + track state label
            kf_pred = self._kf_tracker.predicted_center
            kf_st = self._kf_tracker.state
            if kf_pred is not None and kf_st != ImageSpaceKalmanTracker.UNINITIALIZED:
                px, py = int(kf_pred[0]), int(kf_pred[1])
                # Cyan crosshair at KF predicted centroid
                cv2.drawMarker(annotated_image, (px, py), (255, 255, 0), cv2.MARKER_CROSS, 20, 2)
                # State label near the crosshair
                state_label = f'KF:{kf_st[:4]}'
                if not self._kf_tracker.is_nis_consistent:
                    state_label += ' NIS!'
                cv2.putText(annotated_image, state_label, (px + 12, py - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
            comp_msg = CompressedImage()
            comp_msg.header = msg.header
            comp_msg.format = 'jpeg'
            comp_msg.data = bytes(cv2.imencode('.jpg', annotated_image, [cv2.IMWRITE_JPEG_QUALITY, 75])[1])
            self.detection_pub.publish(comp_msg)
            self._log_annotated_publish_count += 1

        if self.publish_debug_annotated_image and self.camera_info is not None:
            self.publish_ray_debug_image(bgr_image, self.camera_info, local_ray_data, msg.header)

        self._publish_detector_evaluation_packet(
            msg,
            detector_eval_debug,
            infer_ms,
            img_w,
            img_h,
            use_roi,
            roi_x1,
            roi_y1,
            roi_x2,
            roi_y2,
            roi_pred_center,
            best_detection,
            kf_state_before_selection,
            self._kf_tracker.state,
            bool(accepted) if best_detection is not None else None,
            local_ray_data is not None,
        )

        self.frame_counter += 1
        now = time.time()
        if now - self.last_log_time >= 2.0:
            fps = self.frame_counter / (now - self.last_log_time)
            avg_infer_ms = self._log_infer_total_ms / max(1, self.frame_counter)
            mode_summary = self._format_log_window_modes()
            kf_state = self._kf_tracker.state
            esc = f' esc={self._roi_consecutive_misses}' if self._roi_consecutive_misses > 0 else ''
            self.get_logger().info(
                f'Observer node: FPS={fps:.2f}, inference_avg={avg_infer_ms:.1f}ms, '
                f'inference_max={self._log_infer_max_ms:.1f}ms, {mode_summary}{esc}, '
                f'kf={kf_state}, dropped={self._dropped_frames}, annotated_pub={self._log_annotated_publish_count}'
                f'{roi_debug_details}'
            )
            self._reset_log_window(now)

    def _update_roi_tracker(self, det: DetectionObservation, results, stamp_ns: int, roi_x1: int, roi_y1: int) -> None:
        """Update ROI position, box, and smoothed image-plane velocity from the chosen detection."""
        result = results[0]
        boxes_xyxy = result.boxes.xyxy.tolist()
        if not boxes_xyxy:
            return

        bx1, by1, bx2, by2 = min(
            boxes_xyxy,
            key=lambda box: (
                (0.5 * (box[0] + box[2]) + roi_x1 - det.center_u) ** 2
                + (0.5 * (box[1] + box[3]) + roi_y1 - det.center_v) ** 2
            ),
        )

        bx1 += roi_x1
        by1 += roi_y1
        bx2 += roi_x1
        by2 += roi_y1

        new_u = det.center_u
        new_v = det.center_v

        if self._roi_last_center is not None and self._roi_last_stamp_ns > 0:
            dt = (stamp_ns - self._roi_last_stamp_ns) / 1_000_000_000.0
            if dt > 0.001:
                alpha = 0.6
                raw_vel_u = (new_u - self._roi_last_center[0]) / dt
                raw_vel_v = (new_v - self._roi_last_center[1]) / dt
                self._roi_vel_u = alpha * raw_vel_u + (1.0 - alpha) * self._roi_vel_u
                self._roi_vel_v = alpha * raw_vel_v + (1.0 - alpha) * self._roi_vel_v

        self._roi_last_center = (new_u, new_v)
        self._roi_last_bbox = (bx1, by1, bx2, by2)
        self._roi_last_stamp_ns = stamp_ns

    def _record_log_window_frame(self, infer_ms: float, use_roi: bool) -> None:
        self._log_infer_total_ms += infer_ms
        self._log_infer_max_ms = max(self._log_infer_max_ms, infer_ms)
        if use_roi:
            self._log_roi_frames += 1
        else:
            self._log_full_frames += 1

    def _format_log_window_modes(self) -> str:
        return f'modes=ROI:{self._log_roi_frames}/FULL:{self._log_full_frames}'

    def _reset_log_window(self, now: float) -> None:
        self.frame_counter = 0
        self._dropped_frames = 0
        self._log_infer_total_ms = 0.0
        self._log_infer_max_ms = 0.0
        self._log_roi_frames = 0
        self._log_full_frames = 0
        self._log_annotated_publish_count = 0
        self.last_log_time = now

    def _get_tc_projection(
        self,
        now_ns: int,
        camera_matrix: np.ndarray,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Compute the tight-coupling 2D gate from the 3D IMM state.

        Projects the 3D target position (and its covariance) from the global
        odometry frame into the image plane of this observer.

        Architecture Reference:
            Carlson, N.A. (1990). "Federated Square Root Filter for Decentralized
            Parallel Processes." IEEE Trans. Aerospace & Electronic Systems, 26(3).
            → Eq. (12): Justifies the feedback disengagement condition based on
              the information content (covariance trace) of the global filter state.

        Projection Reference:
            Hartley & Zisserman (2004). "Multiple View Geometry in Computer Vision",
            2nd ed., Cambridge. Chapter 6, Eq. (6.14).
            → Analytical Jacobian H_cam for linearised covariance projection.

        Safety Switch (Carlson 1990, Sec. V-B):
            Feedback is disabled when:
              (a) The 3D estimate is stale (age > max_3d_age_ns).
              (b) tr(P_pos) > max_3d_covariance_trace — the 3D filter is coasting
                  after an occlusion and its ellipse would expand to "swallow" any
                  detection on the image plane (the Swallowing Effect).

        Returns:
            (projected_uv, S_prior) if the gate is active and valid,
            None otherwise. 'projected_uv' is the 2D pixel [u, v] of the
            predicted target position; 'S_prior' is the 2×2 innovation
            covariance matrix defining the gating ellipse.
        """
        if not self.tc_enabled or self._tc_latest_odom is None:
            return None

        # --- Safety Switch (Carlson 1990, Sec. V-B): check staleness ---
        age_ns = now_ns - self._tc_latest_odom_stamp_ns
        if age_ns > self.tc_max_3d_age_ns:
            return None

        # Extract 3D position from odometry
        pos = self._tc_latest_odom.pose.pose.position
        p_world_odom = np.array([pos.x, pos.y, pos.z], dtype=np.float64)

        # --- Velocity-Compensated Forward Projection ---
        # The 3D odometry estimate was computed at t_odom, but we are now
        # evaluating at t_now (the timestamp of the current camera frame).
        # The structural latency dt = t_now - t_odom comes from:
        #   • YOLO inference time on the observer node
        #   • ROS message propagation delay
        #   • triangulation_node processing cycle
        # During aggressive observer maneuvers (e.g. sharp pitch), the TF
        # R_cw(t_now) differs from R_cw(t_odom) significantly, causing the
        # TC gate to project a stale world point to the wrong pixel, resulting
        # in spurious rejections of valid detections.
        #
        # Fix: extrapolate p_world to t_now using the IMM-estimated velocity
        # and acceleration published in the twist fields of the odometry msg.
        # This is the same kinematic model used by the IMM-CA filter.
        #   p(t_now) ≈ p(t_odom) + v * dt + 0.5 * a * dt²
        # Reference: Singer (1970) — constant acceleration model; also used
        # as the predict step in Bar-Shalom et al. (2001), Sec. 6.3.
        dt_odom_sec = float(age_ns) / 1e9
        vel = self._tc_latest_odom.twist.twist.linear
        v_world = np.array([vel.x, vel.y, vel.z], dtype=np.float64)
        # [L2-bug1 fix] Use EMA-filtered acceleration estimate from _tc_odom_callback
        # instead of getattr fallback that was always zero.
        a_world = self._tc_accel_estimate
        p_world = (
            p_world_odom
            + v_world * dt_odom_sec
            + 0.5 * a_world * dt_odom_sec * dt_odom_sec
        )

        # Extract positional 3x3 covariance from the 6x6 pose covariance
        # ROS covariance layout: [xx, xy, xz, ..., yy, yz, ..., zz] row-major
        cov6 = np.array(self._tc_latest_odom.pose.covariance, dtype=np.float64).reshape(6, 6)
        P_pos = cov6[:3, :3]

        # Propagate covariance forward by dt using the CV process noise model.
        # Q_dt = sigma_v² * dt² * I  (dominant uncertainty is velocity × time)
        # This inflates the gate appropriately during the prediction horizon,
        # preventing over-confident rejection when the target is moving fast.
        sigma_v_tc = getattr(self, 'tc_sigma_v_inflate', 1.0)  # m/s std dev
        P_pos = P_pos + np.eye(3) * (sigma_v_tc * dt_odom_sec) ** 2

        # --- Safety Switch: check uncertainty magnitude ---
        cov_trace = float(np.trace(P_pos))
        if cov_trace > self.tc_max_3d_covariance_trace:
            return None

        # Project 3D world point into camera frame
        camera_frame = self._resolve_camera_frame(
            self._camera_frame_override or (self.camera_info.header.frame_id if self.camera_info else '')
        )
        if not camera_frame:
            return None

        try:
            stamp_sec = now_ns // 1_000_000_000
            stamp_nsec = now_ns % 1_000_000_000
            from rclpy.time import Time as RclpyTime
            from rclpy.duration import Duration as RclpyDuration
            T = self.tf_buffer.lookup_transform(
                camera_frame,
                self.world_frame_id,
                RclpyTime(seconds=stamp_sec, nanoseconds=stamp_nsec),
                RclpyDuration(seconds=self.tf_timeout_sec),
            )
        except Exception:
            try:
                T = self.tf_buffer.lookup_transform(
                    camera_frame,
                    self.world_frame_id,
                    RclpyTime(),
                    RclpyDuration(seconds=self.tf_timeout_sec),
                )
            except Exception:
                return None

        t = T.transform.translation
        q = T.transform.rotation
        R_cw = quat_to_rotmat(q.x, q.y, q.z, q.w)
        t_cw = np.array([t.x, t.y, t.z], dtype=np.float64)

        # Transform (velocity-compensated) world point to camera frame:
        # p_c = R_cw @ p_world(t_now) + t_cw
        p_cam = R_cw @ p_world + t_cw
        Xc, Yc, Zc = p_cam
        if Zc < 0.1:
            # Target is behind the camera
            return None

        fx = float(camera_matrix[0, 0])
        fy = float(camera_matrix[1, 1])

        # Project onto image plane
        u = fx * Xc / Zc + camera_matrix[0, 2]
        v = fy * Yc / Zc + camera_matrix[1, 2]
        projected_uv = np.array([u, v], dtype=np.float64)

        # --- Analytical Camera Jacobian (Hartley & Zisserman, Ch.6, Eq. 6.14) ---
        # Maps uncertainty from 3D camera frame to 2D pixel plane.
        # H_cam ∈ R^{2×3}
        H_cam = np.array([
            [fx / Zc,      0.0, -fx * Xc / (Zc * Zc)],
            [    0.0, fy / Zc, -fy * Yc / (Zc * Zc)],
        ], dtype=np.float64)

        # Propagate 3D positional covariance into image plane:
        # P_pos is in world frame; transform to camera frame first.
        # P_pos_cam = R_cw @ P_pos @ R_cw^T  (rotation only, ignoring TF uncertainty)
        P_pos_cam = R_cw @ P_pos @ R_cw.T

        # [L2-bug2 fix] R_pixel is now parametric: read from tc_r_pixel_variance (= tc_r_pixel_sigma_px^2).
        # Previously hardcoded to 100.0 (10 px std-dev), which was inconsistent with the
        # tuned kf_sigma_px used by the local 2D KF.  Using the same sigma ensures that
        # the TC gate and the local KF gate share the same measurement noise floor.
        # Reference: Hartley & Zisserman (2004) Ch.6 — projection uncertainty model.
        R_pixel = np.eye(2, dtype=np.float64) * self.tc_r_pixel_variance
        S_prior = H_cam @ P_pos_cam @ H_cam.T + R_pixel

        return projected_uv, S_prior

    def _is_inside_tc_gate(
        self,
        center_u: float,
        center_v: float,
        tc_projection: Tuple[np.ndarray, np.ndarray],
    ) -> bool:
        """Return True if a detection centroid falls inside the 3D-informed gate.

        Computes the Mahalanobis distance between the detection and the
        projected 3D estimate using the innovation covariance S_prior.

        d² = (z - ẑ)ᵀ S⁻¹ (z - ẑ)  ≤ χ²_threshold

        Returns True (inside gate) if d² ≤ tc_gating_chi2_threshold.
        """
        projected_uv, S_prior = tc_projection
        z = np.array([center_u, center_v], dtype=np.float64)
        innovation = z - projected_uv
        try:
            S_inv = np.linalg.inv(S_prior)
        except np.linalg.LinAlgError:
            return True  # Degenerate S → do not veto
        d2 = float(innovation @ S_inv @ innovation)
        return d2 <= self.tc_gating_chi2_threshold

    def extract_best_detection(
        self,
        results,
        msg: Image,
        roi_pred_center: Optional[Tuple[float, float]] = None,
        roi_x1: int = 0,
        roi_y1: int = 0,
        roi_half_side: float = 1.0,
        camera_matrix: Optional[np.ndarray] = None,
        debug_sink: Optional[Dict[str, object]] = None,
        tc_projection: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        depth_arr: Optional[np.ndarray] = None,
    ) -> Optional[DetectionObservation]:
        """Pick the detection used by the node, optionally biasing the score toward the predicted ROI center.

        When tight_coupling is enabled and the 3D IMM filter is confident, each
        candidate that passes the local 2D KF gate is further tested against a
        3D-reprojected Mahalanobis gate (S_prior = H_cam P_3d H_cam^T + R_pixel).
        If ALL candidates are rejected by the TC gate, we fall back to loose coupling
        to avoid deadlock during sensor disagreement.

        When depth_fusion is enabled, a scalar depth gate (Bar-Shalom 2001, Sec. 3.4)
        runs after the TC gate to reject collinear near-occluders.
        """
        collect_debug = debug_sink is not None
        if collect_debug:
            debug_sink.clear()
            debug_sink.update(
                {
                    'raw_total': 0,
                    'class_filtered_count': 0,
                    'prefilter_rejected_count': 0,
                    'ghost_rejected_count': 0,
                    'tc_rejected_count': 0,
                    'depth_rejected_count': 0,
                    'observer_rejected_count': 0,
                    'tc_active': False,
                    'depth_gate_active': False,
                    'candidate_count': 0,
                    'prefilter_candidate_count': 0,
                    'selection_pool': 'none',
                    'selected_index': -1,
                    'detections': [],
                }
            )

        if not results:
            return None

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return None

        names = result.names if result.names is not None else {}
        boxes_xyxy: List[Tuple[float, float, float, float]] = result.boxes.xyxy.tolist()
        confidences: List[float] = result.boxes.conf.tolist()
        classes: List[float] = result.boxes.cls.tolist()
        if collect_debug:
            debug_sink['raw_total'] = len(boxes_xyxy)

        self._ghost_reentry_ready = False
        candidates: List[DetectionObservation] = []
        prefiltered_candidates: List[DetectionObservation] = []
        tc_rejected_candidates: List[DetectionObservation] = []
        ghost_rejected_count = 0
        ghost_recovery_count = 0

        if collect_debug:
            debug_sink['tc_active'] = tc_projection is not None

        # [Layer 4] Observer Self-Rejection: compute the sibling's projected pixel
        # position ONCE per frame, then pick only the SINGLE nearest in-gate raw
        # detection as the veto candidate (Global Nearest Neighbor association
        # within a validation gate — Bar-Shalom et al. 2001; Blackman & Popoli
        # 1999, Ch. 7) instead of blanket-rejecting every detection inside the
        # disc. This matters because gate_px scales as 1/Zc: when the sibling is
        # very close it legitimately projects to a large pixel disc, which would
        # otherwise also swallow an unrelated, distant detection (e.g. the real
        # target far in the background) that merely shares a similar bearing.
        # We know there is exactly ONE sibling drone, so at most one detection
        # per frame can genuinely be it.
        _osr_veto_source_index = -1
        if self.osr_enabled and camera_matrix is not None:
            _osr_gate = self._get_other_observer_gate(stamp_to_ns(msg.header.stamp), camera_matrix)
            if _osr_gate is not None:
                _gu, _gv, _gate_px = _osr_gate
                _best_d2: Optional[float] = None
                for _src_idx, (_box, _conf, _cls_idx) in enumerate(zip(boxes_xyxy, confidences, classes)):
                    if self.target_class_id >= 0 and int(_cls_idx) != self.target_class_id:
                        continue
                    _cu = 0.5 * (_box[0] + _box[2]) + roi_x1
                    _cv = 0.5 * (_box[1] + _box[3]) + roi_y1
                    _du = _cu - _gu
                    _dv = _cv - _gv
                    _d2 = _du * _du + _dv * _dv
                    if _d2 <= _gate_px * _gate_px and (_best_d2 is None or _d2 < _best_d2):
                        _best_d2 = _d2
                        _osr_veto_source_index = _src_idx
                if _best_d2 is None:
                    self.get_logger().info(
                        f'OSR: sibling projected at ({_gu:.0f},{_gv:.0f}) gate_px={_gate_px:.1f} '
                        'but no detection this frame falls inside it.',
                        throttle_duration_sec=2.0,
                    )

        def choose_best_candidate(pool: List[DetectionObservation]) -> Optional[DetectionObservation]:
            if not pool:
                return None

            if roi_pred_center is not None and self.roi_sel_lambda > 0.0:
                pred_u, pred_v = roi_pred_center
                d_max = max(roi_half_side * 0.5, 1.0)

                def score(det: DetectionObservation) -> float:
                    d_pixel = ((det.center_u + roi_x1 - pred_u) ** 2 + (det.center_v + roi_y1 - pred_v) ** 2) ** 0.5
                    return det.confidence - self.roi_sel_lambda * min(d_pixel / d_max, 1.0)

                return max(pool, key=score)

            return max(pool, key=lambda item: item.confidence)

        for source_index, (box, conf, cls_idx) in enumerate(zip(boxes_xyxy, confidences, classes)):
            class_id = int(cls_idx)
            x1, y1, x2, y2 = box
            center_u = 0.5 * (x1 + x2)
            center_v = 0.5 * (y1 + y2)
            class_name = names.get(class_id, str(class_id)) if isinstance(names, dict) else str(class_id)
            bbox_w = float(x2 - x1)
            bbox_h = float(y2 - y1)
            debug_entry = None
            if collect_debug:
                debug_entry = {
                    'source_index': int(source_index),
                    'class_id': int(class_id),
                    'class_name': class_name,
                    'confidence': float(conf),
                    'bbox_xyxy': [
                        float(x1 + roi_x1),
                        float(y1 + roi_y1),
                        float(x2 + roi_x1),
                        float(y2 + roi_y1),
                    ],
                    'center_uv': [float(center_u + roi_x1), float(center_v + roi_y1)],
                    'decision': 'raw',
                }
                debug_sink['detections'].append(debug_entry)

            if self.target_class_id >= 0 and class_id != self.target_class_id:
                if collect_debug:
                    debug_entry['decision'] = 'class_filtered'
                    debug_sink['class_filtered_count'] += 1
                continue

            # [Layer 4] Observer Self-Rejection: reject ONLY the single detection
            # already identified (pre-scan above) as the nearest in-gate match to
            # the sibling's projected position — never the whole disc — so that
            # an unrelated, distant detection sharing the same bearing neighborhood
            # (e.g. the real target far in the background) is never collaterally
            # discarded. Reference: Blackman & Popoli (1999) Ch. 7 (negative object
            # recognition gate) combined with Global Nearest Neighbor association.
            if self.osr_enabled and source_index == _osr_veto_source_index:
                if collect_debug:
                    debug_entry['decision'] = 'observer_rejected'
                    debug_sink['observer_rejected_count'] += 1
                continue

            observation = DetectionObservation(
                stamp_ns=stamp_to_ns(msg.header.stamp),
                center_u=center_u,
                center_v=center_v,
                confidence=float(conf),
                class_id=class_id,
                class_name=class_name,
                bbox_w=bbox_w,
                bbox_h=bbox_h,
                source_index=source_index,
            )

            # Pre-filter using KF Mahalanobis gate if KF is tracking
            if self.kf_enabled and self._kf_tracker.state in (ImageSpaceKalmanTracker.TENTATIVE, ImageSpaceKalmanTracker.CONFIRMED):
                if not self._kf_tracker.evaluate_gate(center_u + roi_x1, center_v + roi_y1, float(conf), bbox_w, bbox_h, camera_matrix):
                    prefiltered_candidates.append(observation)
                    if collect_debug:
                        debug_entry['decision'] = 'kf_prefilter_rejected'
                        debug_sink['prefilter_rejected_count'] += 1
                    continue
            # Ghost spatial filter: when KF is UNINITIALIZED but we recently had a track,
            # reject detections that are clearly far from the last known target position.
            # This prevents trees/false positives from hijacking the M-of-N init gate.
            elif self.kf_enabled and self._kf_tracker.state == ImageSpaceKalmanTracker.UNINITIALIZED:
                if self._kf_tracker.is_ghost_valid(stamp_to_ns(msg.header.stamp)):
                    ghost_u, ghost_v = self._kf_tracker.ghost_center
                    full_center_u = center_u + roi_x1
                    full_center_v = center_v + roi_y1
                    ghost_dist_sq = (full_center_u - ghost_u) ** 2 + (full_center_v - ghost_v) ** 2
                    # Gate radius derived from covariance cap: sqrt(3 * chi2 * (p_max + sigma_px^2))
                    # Generous 3x multiplier allows for drift during the untracked period
                    ghost_gate_radius_sq = 3.0 * self._kf_tracker._chi2_threshold * (
                        self._kf_tracker._p_max + self._kf_tracker._sigma_px ** 2
                    )
                    if ghost_dist_sq > ghost_gate_radius_sq:
                        if self._check_ghost_reentry_gate(
                            full_center_u,
                            full_center_v,
                            float(conf),
                            observation.stamp_ns,
                            roi_pred_center,
                            roi_half_side,
                        ):
                            ghost_recovery_count += 1
                            # Let it fall through to the Tight Coupling gate
                        else:
                            ghost_rejected_count += 1
                        if collect_debug:
                            debug_entry['decision'] = 'ghost_rejected'
                        continue

            # Apply Tight Coupling veto: if the 3D gate is active and this detection
            # falls outside the reprojected ellipse, move it to tc_rejected_candidates.
            # This implements the top-down Mahalanobis gate:
            #   d² = (z-ẑ)ᵀ S_prior⁻¹ (z-ẑ) where S_prior = H_cam P_3d H_cam^T + R_px
            # Reference: Hartley & Zisserman (2004), Ch.6 — Jacobian-based covariance
            # projection; Carlson (1990) — safety-switch for feedback architectures.
            full_u = center_u + roi_x1
            full_v = center_v + roi_y1
            if tc_projection is not None and not self._is_inside_tc_gate(full_u, full_v, tc_projection):
                tc_rejected_candidates.append(observation)
                if collect_debug:
                    debug_entry['decision'] = 'tc_rejected'
                    debug_sink['tc_rejected_count'] += 1
                continue

            # Depth consistency gate (collinear occluder rejection).
            # Runs AFTER 2D-KF and TC gates so those two filters have already
            # removed most false positives; depth is the final disambiguator
            # for near objects that are collinear with the true target.
            # Reference: Bar-Shalom et al. (2001) — scalar gating, Sec. 3.4.
            # This block is completely skipped when df_enabled=False.
            if self.df_enabled and depth_arr is not None:
                full_x1 = x1 + roi_x1  # full-frame bbox coords
                full_y1 = y1 + roi_y1
                full_x2 = x2 + roi_x1
                full_y2 = y2 + roi_y1
                depth_valid, z_meas, sigma_z_meas = self._extract_depth_from_bbox(
                    depth_arr,
                    float(full_x1), float(full_y1),
                    float(full_x2), float(full_y2),
                )
                if depth_valid:
                    _cam_frame = self._resolve_camera_frame(
                        self._camera_frame_override
                        or (self.camera_info.header.frame_id if self.camera_info else '')
                    )
                    # [L3-bug1 fix] Build optical-frame detection ray for geometrically
                    # correct along-ray range prediction (not Z-axis approximation).
                    _detection_ray_optical: Optional[np.ndarray] = None
                    if camera_matrix is not None:
                        _fx = float(camera_matrix[0, 0])
                        _fy = float(camera_matrix[1, 1])
                        _cx = float(camera_matrix[0, 2])
                        _cy = float(camera_matrix[1, 2])
                        _full_cu = 0.5 * (full_x1 + full_x2)
                        _full_cv = 0.5 * (full_y1 + full_y2)
                        _detection_ray_optical = np.array(
                            [(_full_cu - _cx) / _fx, (_full_cv - _cy) / _fy, 1.0],
                            dtype=np.float64,
                        )
                    if _cam_frame and not self._check_depth_gate(
                        z_meas, sigma_z_meas,
                        stamp_to_ns(msg.header.stamp),
                        _cam_frame,
                        detection_ray_optical=_detection_ray_optical,
                    ):
                        if collect_debug:
                            debug_entry['decision'] = 'depth_rejected'
                            debug_sink['depth_rejected_count'] += 1
                        continue

            if collect_debug:
                debug_entry['decision'] = 'candidate'
            candidates.append(observation)

        if collect_debug:
            debug_sink['ghost_rejected_count'] = ghost_rejected_count
            debug_sink['candidate_count'] = len(candidates)
            debug_sink['prefilter_candidate_count'] = len(prefiltered_candidates)

        if not candidates:
            if prefiltered_candidates:
                self.get_logger().info(
                    '[KF] Approximate pre-gate rejected all detections; falling back to exact predict+gate',
                    throttle_duration_sec=1.0,
                )
                selected = choose_best_candidate(prefiltered_candidates)
                if collect_debug:
                    debug_sink['selection_pool'] = 'prefiltered_fallback'
                    if selected is not None:
                        debug_sink['selected_index'] = int(selected.source_index)
                        for item in debug_sink['detections']:
                            if item['source_index'] == selected.source_index:
                                item['decision'] = 'selected_prefilter_fallback'
                                break
                return selected
            if ghost_rejected_count > 0:
                self.get_logger().info(
                    '[KF] Ghost gate rejected all detections while track was UNINITIALIZED',
                    throttle_duration_sec=1.0,
                )
            return None

        if ghost_recovery_count > 0:
            self.get_logger().info(
                '[KF] Re-entry detection admitted after evidence-driven recovery despite ghost mismatch',
                throttle_duration_sec=1.0,
            )

        selected = choose_best_candidate(candidates)
        if collect_debug:
            debug_sink['selection_pool'] = 'candidates'
            if selected is not None:
                debug_sink['selected_index'] = int(selected.source_index)
                for item in debug_sink['detections']:
                    if item['source_index'] == selected.source_index:
                        item['decision'] = 'selected_candidate'
                        break

        return selected

    def draw_predictions(
        self,
        image: np.ndarray,
        results,
        best_detection: Optional[DetectionObservation],
        use_roi: bool = False,
        roi_x1: int = 0,
        roi_y1: int = 0,
        roi_x2: int = 0,
        roi_y2: int = 0,
        kf_tracker: Optional['ImageSpaceKalmanTracker'] = None,
        camera_matrix: Optional[np.ndarray] = None,
        debug_sink: Optional[Dict[str, object]] = None,
        tc_projection: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> np.ndarray:
        """Annotate the image with detection results, colour-coded by pipeline layer.

        Colour legend (detection3 topic):
          - Grey   (140,140,140): Rejected by the 2D image-space Kalman gate  [KF REJ]
          - Red    (  0,  0,220): Passed 2D KF gate but rejected by TC        [TC REJ]
          - Blue   (255,  0,  0): Passed KF+TC but rejected by depth gate     [DEPTH REJ]
          - Yellow (  0,220,220): Rejected by OSR (other observer drone)      [OBS REJ]
          - Orange (  0,165,255): The selected best detection passed all gates [ * ]
          - Green  (  0,255,  0): Other YOLO detection that passed all gates (not selected)
        """
        annotated = image.copy()

        if not results:
            return annotated

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            if use_roi and roi_x2 > roi_x1 and roi_y2 > roi_y1:
                cv2.rectangle(annotated, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 220, 0), 2)
                cv2.putText(
                    annotated, 'ROI', (roi_x1, roi_y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1, cv2.LINE_AA,
                )
            return annotated

        names = result.names if result.names is not None else {}
        boxes_xyxy: List[Tuple[float, float, float, float]] = result.boxes.xyxy.tolist()
        confidences: List[float] = result.boxes.conf.tolist()
        classes: List[float] = result.boxes.cls.tolist()

        # Build a per-source-index decision map from the debug_sink if available.
        # Decision labels are set by extract_best_detection.
        decision_by_idx: Dict[int, str] = {}
        if debug_sink is not None and 'detections' in debug_sink:
            for entry in debug_sink['detections']:
                decision_by_idx[int(entry.get('source_index', -1))] = str(entry.get('decision', 'raw'))

        for source_index, (box, conf, cls_idx) in enumerate(zip(boxes_xyxy, confidences, classes)):
            x1 = int(box[0]) + roi_x1
            y1 = int(box[1]) + roi_y1
            x2 = int(box[2]) + roi_x1
            y2 = int(box[3]) + roi_y1
            class_id = int(cls_idx)
            class_name = names.get(class_id, str(class_id)) if isinstance(names, dict) else str(class_id)
            label = f'{class_name} {conf:.2f}'

            decision = decision_by_idx.get(source_index, 'raw')

            if decision in ('observer_rejected',):
                # Yellow (BGR 0,220,220): rejected by Layer 4 Observer Self-Rejection.
                # The detection centroid fell inside the projected footprint of the
                # other observer drone.  Still drawn so the operator can see what
                # was suppressed by the OSR gate.
                color = (0, 220, 220)
                label += ' [OBS REJ]'
            elif decision in ('tc_rejected',):
                # Red: passed the 2D KF gate but vetoed by the 3D Tight Coupling gate.
                # Still drawn so the operator can see what was suppressed.
                color = (0, 0, 220)
                label += ' [TC REJ]'
            elif decision in ('depth_rejected',):
                # Blue: passed KF + TC but rejected by the depth consistency gate.
                # Indicates a collinear near-occluder that was laterally valid but
                # at a different range than the tracked target.
                color = (255, 0, 0)
                label += ' [DEPTH REJ]'
            elif decision in ('kf_prefilter_rejected',):
                # Grey: rejected by the 2D Kalman gate.
                color = (140, 140, 140)
                label += ' [KF REJ]'
            elif decision in ('selected_candidate', 'selected_prefilter_fallback', 'selected_tc_fallback'):
                # Orange: the detection chosen as the best measurement.
                color = (0, 165, 255)
                label += ' *'
            elif decision in ('class_filtered', 'ghost_rejected'):
                # Dark grey: filtered before entering any gate.
                color = (80, 80, 80)
            else:
                # Green: passed all gates (but not selected because a better one exists).
                # Fallback: re-evaluate KF gate when no debug_sink was populated.
                color = (0, 255, 0)
                if kf_tracker is not None and kf_tracker.state != kf_tracker.UNINITIALIZED:
                    kf_accepted = kf_tracker.evaluate_gate(
                        center_u=0.5 * (box[0] + box[2]) + roi_x1,
                        center_v=0.5 * (box[1] + box[3]) + roi_y1,
                        confidence=conf,
                        bbox_w=box[2] - box[0],
                        bbox_h=box[3] - box[1],
                        camera_matrix=camera_matrix,
                    )
                    if kf_accepted is False:
                        color = (140, 140, 140)
                        label += ' [KF REJ]'
                elif best_detection is not None:
                    cu = 0.5 * (box[0] + box[2]) + roi_x1
                    cv_ = 0.5 * (box[1] + box[3]) + roi_y1
                    if abs(cu - best_detection.center_u) < 1.0 and abs(cv_ - best_detection.center_v) < 1.0:
                        color = (0, 165, 255)
                        label += ' *'

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, self.line_width)
            y_text = y1 - 10 if y1 > 20 else y1 + 20
            cv2.putText(annotated, label, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        # Draw the Tight Coupling projected gate centroid as a cyan diamond marker.
        # This shows the operator exactly where the 3D filter says the target should be.
        if tc_projection is not None:
            projected_uv, _ = tc_projection
            pu, pv = int(round(projected_uv[0])), int(round(projected_uv[1]))
            cv2.drawMarker(annotated, (pu, pv), (255, 200, 0), cv2.MARKER_DIAMOND, 18, 2)
            cv2.putText(
                annotated, 'TC', (pu + 12, pv - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1, cv2.LINE_AA,
            )

        # [Layer 4] Draw the projected other-observer drone position as a magenta circle.
        # Detections inside this circle are suppressed by the OSR gate.  The radius is
        # a visual approximation only; the actual gate uses the 3D body-radius formula.
        osr_proj = getattr(self, '_last_other_observer_proj', None)
        if self.osr_enabled and osr_proj is not None:
            ou, ov = int(round(osr_proj[0])), int(round(osr_proj[1]))
            cv2.circle(annotated, (ou, ov), 20, (255, 0, 255), 2)
            cv2.putText(
                annotated, 'OBS', (ou + 22, ov),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1, cv2.LINE_AA,
            )

        if use_roi and roi_x2 > roi_x1 and roi_y2 > roi_y1:
            cv2.rectangle(annotated, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 220, 0), 2)
            cv2.putText(
                annotated, 'ROI', (roi_x1, roi_y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1, cv2.LINE_AA,
            )

        return annotated

    def compute_and_publish_ray(
        self,
        obs: DetectionObservation,
        depth_arr: Optional[np.ndarray] = None,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Freeze the current detection into a 3D measurement packet.

        When depth_fusion is enabled and a valid depth measurement can be
        extracted from depth_arr, a V2 packet is published carrying bearing
        + range (Bar-Shalom et al. 2001, Sec. 5 — multi-modal sequential update).
        When depth is unavailable or disabled, a V1 (bearing-only) packet is
        published — fully backward-compatible with existing triangulation nodes.
        """
        if self.camera_info is None:
            return None

        fx = float(self.camera_info.k[0])
        fy = float(self.camera_info.k[4])
        cx = float(self.camera_info.k[2])
        cy = float(self.camera_info.k[5])

        ray_optical = normalize(np.array([(obs.center_u - cx) / fx, (obs.center_v - cy) / fy, 1.0], dtype=np.float64))
        camera_frame = self._resolve_camera_frame(self._camera_frame_override or self.camera_info.header.frame_id)

        transform_info = self._lookup_measurement_transform(camera_frame, obs.stamp_ns)
        if transform_info is None:
            return None
        transform, tf_delta_ns, tf_status = transform_info

        t = transform.transform.translation
        q = transform.transform.rotation
        camera_origin_world = np.array([t.x, t.y, t.z], dtype=np.float64)
        r_wc = quat_to_rotmat(q.x, q.y, q.z, q.w)
        ray_world = normalize(r_wc @ ray_optical)

        # Try to extract depth for V2 packet
        pkt_depth_valid = False
        pkt_depth_m = 0.0
        pkt_depth_sigma = 0.0
        if self.df_enabled and depth_arr is not None:
            _dv, _dm, _ds = self._extract_depth_from_bbox(
                depth_arr,
                obs.center_u - obs.bbox_w * 0.5,
                obs.center_v - obs.bbox_h * 0.5,
                obs.center_u + obs.bbox_w * 0.5,
                obs.center_v + obs.bbox_h * 0.5,
            )
            if _dv:
                pkt_depth_valid = True
                pkt_depth_m = _dm
                pkt_depth_sigma = _ds

        # Publish V2 when depth available, V1 otherwise (backward compatible)
        observation_id = self._next_observation_id()
        if pkt_depth_valid:
            packet_bytes = pack_observation_packet_v2(
                image_stamp_ns=obs.stamp_ns,
                observation_id=observation_id,
                confidence=obs.confidence,
                camera_origin_world=camera_origin_world,
                ray_world=ray_world,
                tf_delta_ns=tf_delta_ns,
                tf_status=tf_status,
                depth_valid=True,
                depth_m=pkt_depth_m,
                depth_sigma_m=pkt_depth_sigma,
            )
        else:
            packet_bytes = pack_observation_packet(
                image_stamp_ns=obs.stamp_ns,
                observation_id=observation_id,
                confidence=obs.confidence,
                camera_origin_world=camera_origin_world,
                ray_world=ray_world,
                tf_delta_ns=tf_delta_ns,
                tf_status=tf_status,
            )
        packet_msg = UInt8MultiArray()
        packet_msg.data = packet_bytes
        self.observation_pub.publish(packet_msg)

        return (camera_origin_world, ray_world)

    def publish_ray_debug_image(self, bgr_image, camera_info, local_ray_data, msg_header) -> None:
        """Draw the local frozen ray on the observer image to verify the outgoing packet geometry."""
        if self.debug_annotated_pub is None:
            return

        debug_img = bgr_image.copy()
        camera_frame = self._resolve_camera_frame(self._camera_frame_override or camera_info.header.frame_id)

        try:
            image_time = Time(seconds=msg_header.stamp.sec, nanoseconds=msg_header.stamp.nanosec)
            transform_wc = self.tf_buffer.lookup_transform(camera_frame, self.world_frame_id, image_time)
        except Exception as exc:
            try:
                transform_wc = self.tf_buffer.lookup_transform(camera_frame, self.world_frame_id, Time())
            except Exception:
                self.get_logger().warn(f'No transform available for debug image: {exc}', throttle_duration_sec=2.0)
                return

        rays_to_draw = []
        colors = []
        if local_ray_data is not None:
            origin, ray = local_ray_data
            rays_to_draw.append((origin, ray))
            colors.append((0, 255, 0))

        debug_img = project_ray_on_image(
            debug_img,
            camera_info,
            transform_wc,
            rays_to_draw,
            color_list=colors,
            length=100.0,
        )

        debug_msg = CompressedImage()
        debug_msg.header = msg_header
        debug_msg.format = 'jpeg'
        debug_msg.data = bytes(cv2.imencode('.jpg', debug_img, [cv2.IMWRITE_JPEG_QUALITY, 75])[1])
        self.debug_annotated_pub.publish(debug_msg)

def main() -> None:
    rclpy.init()
    node = ObserverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()