#!/usr/bin/env python3

import collections
import os
import struct
import time
import dataclasses
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
from std_msgs.msg import String, UInt8MultiArray
import json
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


def clip_segment_to_front(p_a, p_b, z_near=0.1):
    """Clip a camera-frame segment against the z=z_near plane. None if fully behind."""
    za, zb = float(p_a[2]), float(p_b[2])
    if za < z_near and zb < z_near:
        return None
    if za < z_near:
        p_a = p_a + (p_b - p_a) * ((z_near - za) / (zb - za))
    elif zb < z_near:
        p_b = p_a + (p_b - p_a) * ((z_near - za) / (zb - za))
    return p_a, p_b


def bearing_error_deg(origin, ray, point):
    """Angle in degrees between a ray and the true bearing origin->point."""
    v = np.asarray(point, dtype=np.float64) - np.asarray(origin, dtype=np.float64)
    denom = float(np.linalg.norm(v) * np.linalg.norm(ray))
    if denom < 1e-9:
        return float('nan')
    return float(np.degrees(np.arccos(np.clip(float(np.dot(v, ray)) / denom, -1.0, 1.0))))


def bearing_error_az_el_deg(origin, ray, point):
    """Bearing error split into (azimuth, elevation), degrees, in the north-aligned world frame.

    Azimuth is measured in the horizontal plane, elevation against it. A drone heading bias
    lands entirely in azimuth; a Z-datum or camera-pitch fault lands entirely in elevation.
    """
    v = np.asarray(point, dtype=np.float64) - np.asarray(origin, dtype=np.float64)
    r = np.asarray(ray, dtype=np.float64)
    az = np.degrees(np.arctan2(r[1], r[0]) - np.arctan2(v[1], v[0]))
    az = (az + 180.0) % 360.0 - 180.0
    el = np.degrees(
        np.arctan2(r[2], np.hypot(r[0], r[1])) - np.arctan2(v[2], np.hypot(v[0], v[1]))
    )
    return float(az), float(el)


def _bearing_az_el_deg(origin, point):
    """Absolute (azimuth, elevation) of point seen from origin, degrees, world frame."""
    v = np.asarray(point, dtype=np.float64) - np.asarray(origin, dtype=np.float64)
    return (float(np.degrees(np.arctan2(v[1], v[0]))),
            float(np.degrees(np.arctan2(v[2], np.hypot(v[0], v[1])))))


def yaw_deg(q) -> float:
    """Heading of a quaternion's body x-axis in the world XY plane, degrees.

    This is the component a magnetometer sets, and the one that differs between two drones
    sitting parallel on the ground. Roll/pitch are deliberately dropped: they come from the
    IMU, not the compass.
    """
    rot = quat_to_rotmat(q.x, q.y, q.z, q.w)
    return float(np.degrees(np.arctan2(rot[1, 0], rot[0, 0])))


def derotate_bearing(ray, az_deg, el_deg):
    """Remove a constant (azimuth, elevation) bias from a world-frame ray.

    Exact inverse of what bearing_error_az_el_deg measures, so a measured bias can be pasted
    back as a correction. Azimuth is applied about world Z, which is where a magnetometer
    heading error lives -- NOT about the body z of camera_tf.yaw_offset, which differs by the
    drone's roll/pitch (6.6 deg of roll on uav8 in the v5 bag).

    The observer's position is deliberately untouched: it comes from RTK, which a miscalibrated
    compass does not affect. Only the pointing direction is wrong.
    """
    if not az_deg and not el_deg:
        return ray
    r = np.asarray(ray, dtype=np.float64)
    az = np.arctan2(r[1], r[0]) - np.radians(az_deg)
    el = np.arctan2(r[2], np.hypot(r[0], r[1])) - np.radians(el_deg)
    c = np.cos(el)
    return np.array([c * np.cos(az), c * np.sin(az), np.sin(el)], dtype=np.float64)


def rebias_about(cam_pos, origin, ray, az_deg, el_deg):
    """Put a corrected world ray back into the biased world one camera's tf still believes in.

    derotate_bearing takes the compass error OUT of the rays, but a tf lookup still hands back
    the biased camera pose, so projecting a corrected ray with it draws the ray off by the full
    bias. Re-applying the bias here is what makes the two agree again.

    The rotation is about the CAMERA's position, not the world origin: the bias rotates the
    camera's attitude and leaves RTK positions where they are, so a point p must come back as
    cam_pos + Rz(az)*(p - cam_pos). Returns the (origin, ray) pair ready to project.
    """
    if not az_deg and not el_deg:
        return origin, ray
    offset = np.asarray(origin, dtype=np.float64) - np.asarray(cam_pos, dtype=np.float64)
    dist = float(np.linalg.norm(offset))
    if dist > 0.0:
        # derotate_bearing normalises, so carry the length across by hand.
        offset = derotate_bearing(offset, -az_deg, -el_deg) * dist
    return cam_pos + offset, derotate_bearing(ray, -az_deg, -el_deg)


def fit_bias_offsets(samples):
    """Constant (az, el) bias of one observer, fitted on the first half of its samples.

    Returns (az_fit, el_fit, az_residual, el_residual, n) in degrees, or None if too few
    samples. Fitting and scoring on the same frames would be circular, so the split is by
    time: the offsets come from the early half and the residual is measured on frames the
    fit never saw. A residual near zero means the bias really is a constant of the platform
    (a miscalibrated compass) rather than drift or a bad fit.

    Median, not mean: a handful of false detections would drag a mean and leave the offset
    describing the outliers instead of the hardware.
    """
    if len(samples) < 4:
        return None
    ordered = sorted(samples)
    half = len(ordered) // 2
    az_fit = float(np.median([s[1] for s in ordered[:half]]))
    el_fit = float(np.median([s[2] for s in ordered[:half]]))
    az_res = float(np.median([s[1] for s in ordered[half:]])) - az_fit
    el_res = float(np.median([s[2] for s in ordered[half:]])) - el_fit
    return az_fit, el_fit, az_res, el_res, len(ordered)


def metric_or_angular(d_el_deg, rng_m):
    """Score 'constant degrees' against 'constant metres' as explanations of an elevation error.

    A camera that really points wrong holds its DEGREES at every range. A ray whose ORIGIN
    (or whose target's GT) sits at the wrong height holds its METRES, and its angle decays as
    1/range. At a single range the two are indistinguishable, which is how a fitted el_deg
    ends up not transferring to the next flight.

    Both are scored on the same axis -- metres of vertical miss at the target -- so the
    residuals compare directly. Returns (med_deg, med_m, res_ang_m, res_met_m, r_lo, r_hi).
    """
    d_el_deg = np.asarray(d_el_deg, dtype=np.float64)
    rng_m = np.asarray(rng_m, dtype=np.float64)
    med_deg = float(np.median(d_el_deg))
    miss_m = np.radians(d_el_deg) * rng_m
    med_m = float(np.median(miss_m))
    return (med_deg, med_m,
            float(np.median(np.abs(miss_m - np.radians(med_deg) * rng_m))),
            float(np.median(np.abs(miss_m - med_m))),
            float(rng_m.min()), float(rng_m.max()))


def verdict_metric_or_angular(res_ang, res_met, r_lo, r_hi, med_m, what):
    """Name the winning model, or refuse to when the range span cannot separate them."""
    if r_hi < 1.5 * r_lo:
        return f'range span too narrow to tell them apart (need r_hi > 1.5*r_lo)'
    if res_met < 0.7 * res_ang:
        return f'METRIC: {what} origin/GT height off by {-med_m:+.2f} m, an angle will not fix it'
    if res_ang < 0.7 * res_met:
        return f'ANGULAR: {what} really points wrong, an el_deg correction is the right shape'
    return 'inconclusive'


def differential_bias(s1, s2, max_dt_ns):
    """Median (obs2 - obs1) az/el error, matching samples by nearest timestamp.

    Differencing the two observers cancels any fault they SHARE -- above all the GT's own
    altitude error, since a baseline much shorter than the range puts both at nearly the
    same distance so a metric GT error subtracts out. What survives is the relative geometry
    of the two observers.

    That survivor has two possible homes, and they are NOT interchangeable:

      angular -- uav8's camera really points higher (mount, or a bad attitude solution).
                 The elevation discrepancy is then constant in DEGREES at every range.
      metric  -- uav8's ray ORIGIN sits at the wrong height, because its world_origin z
                 datum is set at its own take-off and differs from uav81's. The discrepancy
                 is then constant in METRES and its angle shrinks as 1/range.

    Both look identical at a single range, which is why an el_deg fitted on one flight does
    not transfer to the next. So both models are scored on the same axis (metres of vertical
    miss at the target) and the residuals returned side by side: whichever residual is
    smaller names the fault, provided the range span is wide enough to tell them apart.

    Returns (n, d_az, d_el_deg, d_el_m, res_ang_m, res_met_m, r_lo, r_hi) or None.
    """
    if len(s1) < 30 or len(s2) < 30:
        return None
    a = np.array(s1, dtype=np.float64)
    b = np.array(s2, dtype=np.float64)
    order_a = np.argsort(a[:, 0])
    a = a[order_a]
    j = np.clip(np.searchsorted(a[:, 0], b[:, 0]), 1, len(a) - 1)
    near = np.where(np.abs(b[:, 0] - a[j - 1, 0]) < np.abs(b[:, 0] - a[j, 0]), j - 1, j)
    ok = np.abs(b[:, 0] - a[near, 0]) <= max_dt_ns

    # The cancellation only works while both observers sit at nearly the same range: a metric
    # GT fault subtends different angles at different distances, so on a dispersed formation
    # it stops subtracting out and leaks into the difference as a fake camera bias. On the v3
    # flight the ratio runs 0.53..2.08, so this is not a hypothetical.
    ratio = b[:, 5] / np.maximum(a[near, 5], 1e-6)
    ok &= (ratio > 0.8) & (ratio < 1.25)
    if int(ok.sum()) < 30:
        return None

    d_az = b[ok, 1] - a[near[ok], 1]
    d_el = b[ok, 2] - a[near[ok], 2]
    med_el, med_m, res_ang, res_met, r_lo, r_hi = metric_or_angular(d_el, b[ok, 5])

    return (int(ok.sum()), float(np.median(d_az)), med_el, med_m,
            res_ang, res_met, r_lo, r_hi)


def fit_field_slope(samples):
    """Where does an observer's bearing error live: the mount, the optics, or the attitude?

    Each fault class leaves a different signature in how the error varies across the field of
    view, so both error axes are regressed against BOTH true bearing axes at once:

        err_el = A*el_true + B*az_true + drift*t + c
        err_az = C*el_true + D*az_true + drift*t + c

    - A camera bolted on crooked (a fixed pitch or yaw offset) rotates the whole ray bundle
      rigidly: every slope 0, error in the constant.
    - Wrong focal length or principal point scales the error with the distance off-axis, so it
      shows on the DIAGONAL terms A and D.
    - A roll about the optical axis moves a point perpendicular to its offset from image centre,
      turning horizontal distance into vertical error. That is the CROSS term B (and -C), and it
      is the one fault that leaves the detector's box perfectly on target in its own image while
      the world-frame ray still misses. B ~= the roll angle in radians.

    Two traps this has to dodge:

    - The target crosses the field of view WHILE time passes, so a slow drift (in the compass,
      or in the target's own odometry) masquerades as a field-of-view slope. Fitting time as its
      own regressor is what separates them; when the target sweeps monotonically the two are
      collinear and NOTHING can separate them, which is what `corr` is for. Above ~0.9 the slope
      is not identifiable and must not be acted on.
    - Regressing against the measured bearing instead of the true one shares the noise between
      the two axes and manufactures a slope out of it, so the regressor is always the true one.

    Valid only while the camera is static (v5 has the observers on a table), where the true world
    bearing differs from the image position by a fixed rotation. On a moving observer the
    body-frame direction has to be used instead.

    Returns a dict per error axis with `d_el`/`d_az` [deg/deg], `drift` [deg/min], `lo`/`hi`
    [deg], plus the regressor spans and their collinearity with time. A slope means nothing
    without span: no leverage, no conclusion.
    """
    if len(samples) < 8:
        return None
    a = np.asarray(samples, dtype=np.float64)
    if a.shape[1] < 5:
        return None
    az_err, el_err, az_true, el_true = a[:, 1], a[:, 2], a[:, 3], a[:, 4]
    t_min = (a[:, 0] - a[:, 0].min()) / 6e10          # ns -> minutes

    el_span, az_span = float(np.ptp(el_true)), float(np.ptp(az_true))
    if el_span <= 1e-6 or az_span <= 1e-6 or t_min.max() <= 1e-9:
        return None
    design = np.column_stack([el_true, az_true, t_min, np.ones_like(el_err)])

    def _axis(err):
        c = np.linalg.lstsq(design, err, rcond=None)[0]
        return dict(d_el=float(c[0]), d_az=float(c[1]), drift=float(c[2]),
                    lo=float(err.min()), hi=float(err.max()))

    return {
        'el': _axis(el_err), 'az': _axis(az_err), 'n': len(a),
        'el_span': el_span, 'az_span': az_span,
        'corr_el_t': abs(float(np.corrcoef(el_true, t_min)[0, 1])),
        'corr_az_t': abs(float(np.corrcoef(az_true, t_min)[0, 1])),
    }


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

        # Clip the segment to the part in front of the camera instead of dropping it.
        # The remote observer's ray starts at the OTHER drone, which is almost always
        # behind/outside this camera: skipping on start_z<=0 made the remote ray invisible
        # in every frame, which is exactly the case we need to see.
        clipped = clip_segment_to_front(point_start_camera, point_end_camera)
        if clipped is None:
            continue
        point_start_camera, point_end_camera = clipped

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

def unpack_observation_packet(msg: UInt8MultiArray) -> Optional[Tuple[int, int, float, np.ndarray, np.ndarray, int, int, bool, float, float]]:
    """Unpack a compact observer measurement packet from the network topic.
    Returns:
        stamp_ns, observation_id, confidence, origin, ray, tf_delta_ns, tf_status, depth_valid, depth_m, depth_sigma_m
    """
    packet_bytes = bytes(msg.data)
    if len(packet_bytes) == OBSERVATION_PACKET_V2_STRUCT.size:
        unpacked = OBSERVATION_PACKET_V2_STRUCT.unpack(packet_bytes)
        stamp_ns, observation_id, confidence = unpacked[:3]
        origin = np.array(unpacked[3:6], dtype=np.float64)
        ray = np.array(unpacked[6:9], dtype=np.float64)
        tf_delta_ns = int(unpacked[9])
        tf_status = int(unpacked[10])
        depth_valid = bool(unpacked[11])
        depth_m = float(unpacked[12])
        depth_sigma_m = float(unpacked[13])
        return int(stamp_ns), int(observation_id), float(confidence), origin, ray, tf_delta_ns, tf_status, depth_valid, depth_m, depth_sigma_m
    elif len(packet_bytes) == OBSERVATION_PACKET_STRUCT.size:
        unpacked = OBSERVATION_PACKET_STRUCT.unpack(packet_bytes)
        stamp_ns, observation_id, confidence = unpacked[:3]
        origin = np.array(unpacked[3:6], dtype=np.float64)
        ray = np.array(unpacked[6:9], dtype=np.float64)
        tf_delta_ns = int(unpacked[9])
        tf_status = int(unpacked[10])
        return int(stamp_ns), int(observation_id), float(confidence), origin, ray, tf_delta_ns, tf_status, False, 0.0, 0.0
    else:
        return None


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


def extract_position_covariance(
    odom_msg: Odometry,
    fallback_variance: float = 0.01,
    logger = None,
    observer_label: str = "Unknown",
) -> np.ndarray:
    """Extract the 3x3 translational position covariance from a nav_msgs/Odometry message.

    The ``pose.covariance`` field is a flat 36-element array representing a 6x6
    row-major matrix ordered as (x, y, z, roll, pitch, yaw).  We extract the
    upper-left 3x3 block corresponding to translational degrees of freedom.

    If the covariance is zero (e.g. not published by the driver) or nearly
    degenerate, a scalar ``fallback_variance * I_3`` is returned to prevent
    the Jacobian term from vanishing while still being conservative.

    Reference: REP-103 (ROS coordinate conventions), nav_msgs/Odometry.msg.

    Args:
        odom_msg:         The odometry message from which to extract covariance.
        fallback_variance: Isotropic variance (m²) used when covariance is
                          unavailable or non-positive-definite.
        logger:           Optional rclpy logger for fallback warnings.
        observer_label:   Name of the observer for log context.

    Returns:
        P_pos (3×3 float64 ndarray): Translational position covariance.
    """
    raw = np.array(odom_msg.pose.covariance, dtype=np.float64).reshape(6, 6)
    P_pos = raw[:3, :3].copy()

    # Fallback: if the diagonal is all-zeros the driver does not publish covariance.
    if np.all(np.abs(np.diag(P_pos)) < 1e-12):
        if logger is not None:
            logger.warn(
                f'\033[95m[odom_uncertainty] {observer_label} Odometry covariance is ZERO. '
                f'Using fallback {fallback_variance} m^2\033[0m',
                throttle_duration_sec=5.0
            )
        return np.eye(3, dtype=np.float64) * fallback_variance

    # Enforce symmetry (numerical rounding in serialisation).
    P_pos = 0.5 * (P_pos + P_pos.T)

    # Enforce positive-definiteness by clamping eigenvalues.
    eigvals = np.linalg.eigvalsh(P_pos)
    if np.any(eigvals <= 0.0):
        P_pos += (abs(eigvals.min()) + 1e-9) * np.eye(3, dtype=np.float64)

    return P_pos


def compute_bearing_jacobian_wrt_observer(
    p_obs: np.ndarray,
    p_target: np.ndarray,
    min_distance: float = 0.5,
) -> np.ndarray:
    """Jacobian of the unit bearing vector b = (p_t - p_obs)/‖p_t - p_obs‖
    with respect to the observer position p_obs.

    Derivation (chain rule on the unit-norm map):

        b(p_obs) = -(p_obs - p_t) / d,   d = ‖p_t - p_obs‖

        ∂b/∂p_obs = -(1/d)(I₃ - b bᵀ)

    This is the standard first-order sensitivity of a bearing measurement to
    errors in the sensor position, as used in decentralised target tracking
    literature (e.g. Zhao et al., 2013; Olfati-Saber, 2006).

    The term (I₃ - b bᵀ) is the projection onto the plane perpendicular to b,
    so only cross-bearing displacements of the observer affect the bearing.
    Motions along b (radial) cancel at first order — consistent with bearing-
    only observability theory.

    Args:
        p_obs:        Observer 3D position in world frame (3,).
        p_target:     Target 3D position in world frame (3,).
        min_distance: Clamp on ‖p_t - p_obs‖ to prevent singularity (m).

    Returns:
        J (3×3 float64 ndarray): Jacobian ∂b/∂p_obs.  Returns zero matrix
        for degenerate geometry (target and observer co-located).
    """
    diff = p_target - p_obs
    d = float(np.linalg.norm(diff))

    if d < 1e-6:
        # Fully degenerate: observer and target are co-located.
        return np.zeros((3, 3), dtype=np.float64)

    d_clamped = max(d, float(min_distance))
    b = diff / d  # unit bearing vector
    # J = -(1/d_clamped) * (I - b bᵀ)
    J = -(1.0 / d_clamped) * (np.eye(3, dtype=np.float64) - np.outer(b, b))
    return J


def _validate_covariance_matrix(
    R: np.ndarray,
    name: str,
    logger=None,
    eps: float = 1e-9,
) -> np.ndarray:
    """Verify symmetry and positive-definiteness of a covariance matrix.

    If either condition is violated the matrix is repaired in-place (symmetrised
    and shifted) and a ROS WARNING is emitted so the issue is visible in logs
    without crashing the node.

    Args:
        R:      Input matrix to validate.
        name:   Human-readable label for log messages.
        logger: Optional rclpy logger (node.get_logger()).
        eps:    Minimum eigenvalue shift to restore PD-ness.

    Returns:
        Validated (and possibly repaired) matrix.
    """
    # Symmetry check
    sym_err = float(np.max(np.abs(R - R.T)))
    if sym_err > 1e-6:
        if logger is not None:
            logger.warn(
                f'[odom_uncertainty] {name} not symmetric (max_asym={sym_err:.2e}); enforcing symmetry.',
                throttle_duration_sec=5.0,
            )
        R = 0.5 * (R + R.T)

    # Positive-definiteness check via eigenvalues
    eigvals = np.linalg.eigvalsh(R)
    min_eig = float(eigvals.min())
    if min_eig <= 0.0:
        shift = abs(min_eig) + eps
        if logger is not None:
            logger.warn(
                f'[odom_uncertainty] {name} not positive-definite (min_eig={min_eig:.2e}); '
                f'adding {shift:.2e}*I.',
                throttle_duration_sec=5.0,
            )
        R = R + shift * np.eye(R.shape[0], dtype=np.float64)

    return R

# World "up" used to define the azimuth direction of a bearing ray.
_WORLD_UP = np.array([0.0, 0.0, 1.0], dtype=np.float64)

# Below this the ray is within ~0.06 deg of vertical and (z x d) is numerically
# meaningless: azimuth is undefined there, so an arbitrary horizontal axis is used.
_VERTICAL_RAY_EPS = 1e-6


def bearing_frame(direction: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Local azimuth/elevation basis of a bearing ray, in the world frame.

    Given a unit bearing ``d`` with elevation ``el = asin(d_z)``:

        a = normalise(z_world x d)     horizontal, tangent to the azimuth circle
        e = d x a                      completes the right-handed triad

    ``a`` and ``e`` are orthonormal and span the plane perpendicular to ``d`` --
    exactly the plane in which a bearing error displaces the target.  Moving the
    ray by ``d_az`` in azimuth displaces the target by ``r * cos(el) * d_az``
    along ``a``; moving it by ``d_el`` displaces it by ``r * d_el`` along ``e``.
    The ``cos(el)`` factor is the Jacobian of the spherical parametrisation and
    is returned so the caller can apply it.

    Degenerate case: for a near-vertical ray azimuth is not defined; world X is
    used as the auxiliary axis so the triad stays orthonormal (with cos(el) ~ 0
    the azimuth term is annihilated anyway).
    """
    d = np.asarray(direction, dtype=np.float64)
    a = np.cross(_WORLD_UP, d)
    n = float(np.linalg.norm(a))
    if n < _VERTICAL_RAY_EPS:
        a = np.cross(np.array([1.0, 0.0, 0.0]), d)
        n = float(np.linalg.norm(a))
        if n < _VERTICAL_RAY_EPS:          # d is not unit-norm: cannot recover
            return np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]), 1.0
    a = a / n
    e = np.cross(d, a)
    cos_el = float(np.clip(np.hypot(d[0], d[1]), 0.0, 1.0))   # = cos(asin(d_z))
    return a, e, cos_el


def bearing_noise_cov(
    direction: np.ndarray,
    var_az: float,
    var_el: float,
    range_scale: float,
) -> np.ndarray:
    """Anisotropic visual noise covariance V_vis (3x3, world frame) of one ray.

    Replaces the isotropic ``var_px * d^2 * I_3`` by a covariance with distinct
    azimuth and elevation variances::

        V_vis = range_scale * ( var_az * cos^2(el) * a a^T      (azimuth axis)
                              + var_el                * e e^T   (elevation axis)
                              + var_r                 * d d^T ) (radial filler)

    with ``(a, e, cos_el)`` from :func:`bearing_frame`.  The three axes are
    orthonormal, so this *is* the eigen-decomposition of ``V_vis``: the
    eigenvalues are literally the per-axis variances, in m^2 after multiplication
    by ``range_scale = d_i^2``.

    Frame check (verified numerically, see test_anisotropic_bearing_noise.py): a
    yaw error about world z lands on ``a`` with weight cos(el) and on ``e`` with
    weight EXACTLY zero; a body-pitch error lands on ``e`` with weight one and on
    ``a`` with weight exactly zero.  The only cross-term is a tilt error along the
    line of sight, which leaks into azimuth with factor sin(el) -- negligible for a
    target at the observer's altitude, 34% at el = 20 deg.

    Why this matters physically: on this airframe the camera is bolted to the FCU
    with identity rotation, so attitude error enters the bearing 1:1.  Yaw error
    is dominated by the dual-antenna RTK heading baseline (sigma ~ sigma_rel / L,
    typically ~1 deg for L = 0.55 m) while pitch/roll error is dominated by the
    per-boot level reference latched at power-up (measured 1.5-3.9 deg on the
    22-7 bags).  These two budgets differ by a factor of a few, and an isotropic
    R cannot express that: it must inflate the good axis or shrink the bad one.

    The radial term is a rank-completing regulariser -- a bearing carries no
    range information -- set to the mean of the two angular variances so V stays
    well conditioned.  It is *exactly* cancelled by the outer projectors Pi_i in
    the Fisher accumulation of :func:`triangulate_wls`, so its value never
    reaches ``R_out``.
    """
    a, e, cos_el = bearing_frame(direction)
    d = np.asarray(direction, dtype=np.float64)
    var_r = 0.5 * (float(var_az) + float(var_el))
    V = (
        float(var_az) * (cos_el ** 2) * np.outer(a, a)
        + float(var_el) * np.outer(e, e)
        + var_r * np.outer(d, d)
    )
    return float(range_scale) * V


def triangulate_wls(
    origin1: np.ndarray,
    dir1: np.ndarray,
    P_odom1: Optional[np.ndarray],
    origin2: np.ndarray,
    dir2: np.ndarray,
    P_odom2: Optional[np.ndarray],
    var_px: float,
    use_odom_uncertainty: bool = True,
    distance_dependent_noise: bool = True,
    odom_max_variance: float = 1.0,
    anisotropic_bearing_noise: bool = False,
    bearing_var_az: float = 0.0,   # TOTAL azimuth variance   [rad^2] (yaw + detector)
    bearing_var_el: float = 0.0,   # TOTAL elevation variance [rad^2] (tilt + detector)
    prior_target_pos: Optional[np.ndarray] = None,
    normalize_crlb: bool = True,
    logger=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Optimal bearing-only triangulation via Weighted Least Squares (Errors-In-Variables).

    Jointly computes the Maximum-Likelihood Estimate of the target position
    ``p_hat`` and its Cramér-Rao Lower Bound covariance ``R_out`` (3x3) from
    two bearing rays, accounting for both visual measurement noise and observer
    odometry uncertainty.

    The algorithm is a two-step pseudo-linear estimator:

    Step 1 — Bootstrap / Instrumental Variable:
        Uses an Instrumental Variable (IV) approach to estimate distance and
        break the Pseudo-Linear Estimator (PLE) bias (Doğançay 2006). If a
        prior target position is provided (from IMM history), it is used directly
        to compute the distance, ensuring WLS weights are independent of current
        bearing noise. If missing, falls back to an Orthogonal Vectors Estimator (OLS).

    Step 2 — Weighted Least Squares:
        Build the per-ray composite noise covariance::

            V_i = (var_px * d_i^2 * I_3)  +  Pi_i @ P_odom_i @ Pi_i.T

        where ``Pi_i = I - outer(dir_i, dir_i)`` is the orthogonal projector
        onto the plane perpendicular to the bearing (Doğançay 2015, Sec. II).
        Accumulate the Fisher Information Matrix::

            Lambda = Pi_1.T @ inv(V_1) @ Pi_1  +  Pi_2.T @ inv(V_2) @ Pi_2

        This equals A.T @ W @ A where A = [Pi_1; Pi_2] and W = block_diag(inv(V_1), inv(V_2)).
        The output covariance and position follow from::

            R_out  = inv(Lambda)          # CRLB (Bishop et al. 2010, Sec. III)
            p_hat  = R_out @ (Pi_1.T @ inv(V_1) @ Pi_1 @ c1
                            + Pi_2.T @ inv(V_2) @ Pi_2 @ c2)

    NOTE on I_3 vs Pi_i for the visual term:
        Using ``var_px * d_i^2 * I_3`` (isotropic sphere) instead of the
        theoretically exact ``var_px * d_i^2 * Pi_i`` (rank-2 disk) injects a
        tiny fictitious variance along the ray direction. This guarantees that
        ``V_i`` is full-rank and stably invertible even when
        ``use_odom_uncertainty=False``. The spurious radial component is
        *exactly* cancelled by the outer ``Pi_i`` projectors in the Lambda
        accumulation step, so the final ``R_out`` is unaffected. This is a
        standard Tikhonov-like regularisation for rank-deficient EIV systems.

    References:
        - Doğançay, K. (2005). Bearings-only target localization using total
          least squares. *Signal Processing*, 85(9), 1695-1710.
          doi:10.1016/j.sigpro.2005.03.007
        - Doğançay, K. (2015). 3D Pseudolinear Target Motion Analysis From
          Angle Measurements. *IEEE Trans. Signal Process.*, 63(6), 1570-1580.
          doi:10.1109/TSP.2015.2399869
        - Xu, S., Doğançay, K., & Hmam, H. (2017). Distributed pseudolinear
          estimation and UAV path optimization for 3D AOA target tracking.
          *Signal Processing*, 133, 64-78. doi:10.1016/j.sigpro.2016.10.012
        - Bishop, A. N., et al. (2010). Optimality analysis of sensor-target
          localization geometries. *Automatica*, 46(3), 479-492.
          doi:10.1016/j.automatica.2009.12.003

    Args:
        origin1/2:              3-D world-frame positions of the two observer cameras (3,).
        dir1/2:                 Unit bearing vectors toward the target (3,).
        P_odom1/2:              3x3 observer position covariance (or None).
        var_px:                 Angular variance seed [rad^2 or m^2/m^2]; the
                                ``kf_measurement_noise`` parameter from YAML.
        use_odom_uncertainty:   When True, include P_odom_i in V_i.
        distance_dependent_noise: When True, visual sigma = var_px * d_i^2.
        anisotropic_bearing_noise: When True the visual term uses separate
                                azimuth/elevation variances (see
                                :func:`bearing_noise_cov`) instead of the
                                isotropic ``var_px * I_3``.  ``var_px`` is
                                then unused for the visual term.
        bearing_var_az:         Azimuth (yaw-like) angular variance [rad^2].
                                Only read when anisotropic_bearing_noise=True.
        bearing_var_el:         Elevation (pitch-like) angular variance [rad^2].
                                Only read when anisotropic_bearing_noise=True.
        odom_max_variance:      Diagonal clamp [m^2] applied to P_odom before
                                projection. Prevents barometric Z-blowup
                                (e.g. sim publishes sigma_z ~ 35 m) from
                                making V_i ill-conditioned.
        prior_target_pos:       Optional prior target position for IV estimation.
        logger:                 Optional rclpy logger.

    Returns:
        p_hat (3,):  MLE target position in world frame.
        R_out (3x3): Symmetric PD measurement covariance (CRLB) → use as R_k
                     in the IMM-KF update step.

    Raises:
        ValueError: If the bearing lines are nearly parallel (observability
                    check fails) or if the linear system is singular.
    """
    I3 = np.eye(3, dtype=np.float64)

    # ── Observability check ──────────────────────────────────────────────────
    cross = np.cross(dir1, dir2)
    sin_angle = float(np.linalg.norm(cross))
    if sin_angle < 0.087:  # sin(5°) ≈ 0.087
        raise ValueError(
            f'[triangulate_wls] Bearing lines nearly parallel '
            f'(sin_angle={sin_angle:.4f} < 0.087). Observability lost.'
        )

    # ── Orthogonal projectors  Pi_i = I - dir_i * dir_i^T ───────────────────
    Pi1 = I3 - np.outer(dir1, dir1)   # (3x3) rango 2
    Pi2 = I3 - np.outer(dir2, dir2)

    # ── Step 1: Bootstrap / Instrumental Variable per stima distanza ─────────
    if prior_target_pos is not None:
        # Metodo IV-WLS: usiamo la predizione passata per rompere il PLE bias
        # Costo computazionale quasi nullo, stabilità geometrica garantita.
        d1 = float(max(np.linalg.norm(prior_target_pos - origin1), 0.1))
        d2 = float(max(np.linalg.norm(prior_target_pos - origin2), 0.1))
    else:
        # Fallback OLS classico per la prima misurazione assoluta o reset
        A_boot = np.vstack([Pi1, Pi2])                          # (6, 3)
        b_boot = np.concatenate([Pi1 @ origin1, Pi2 @ origin2])  # (6,)
        ATA = A_boot.T @ A_boot   # (3, 3)
        ATb = A_boot.T @ b_boot   # (3,)
        try:
            p_hat_0 = np.linalg.solve(ATA, ATb)
        except np.linalg.LinAlgError:
            # Extremely degenerate geometry: fall back to midpoint
            p_hat_0 = 0.5 * (origin1 + origin2)
        d1 = float(max(np.linalg.norm(p_hat_0 - origin1), 0.1))
        d2 = float(max(np.linalg.norm(p_hat_0 - origin2), 0.1))

    # ── Step 2: Build per-ray composite noise covariance V_i ─────────────────
    # Visual term: isotropic I_3 used instead of Pi_i for rank stability
    # (spurious radial component is cancelled by outer Pi projectors in Lambda).
    d_scale_1 = (d1 ** 2) if distance_dependent_noise else 1.0
    d_scale_2 = (d2 ** 2) if distance_dependent_noise else 1.0

    if anisotropic_bearing_noise:
        # Separate azimuth / elevation variances: the ray's uncertainty is an
        # elliptical cone, not a circular one.  Rank is completed along the
        # radial axis inside bearing_noise_cov (cancelled by Pi_i below).
        V1 = bearing_noise_cov(dir1, bearing_var_az, bearing_var_el, d_scale_1)
        V2 = bearing_noise_cov(dir2, bearing_var_az, bearing_var_el, d_scale_2)
    else:
        V1 = float(var_px) * d_scale_1 * I3   # Visual-only base
        V2 = float(var_px) * d_scale_2 * I3

    if use_odom_uncertainty:
        # Clamp diagonal of P_odom to odom_max_variance before projection
        # to neutralise barometric Z-blowup from the MRS simulator.
        def _clamp_and_project(P_odom: Optional[np.ndarray], Pi: np.ndarray) -> np.ndarray:
            if P_odom is None:
                return np.zeros((3, 3), dtype=np.float64)
            P_c = P_odom.copy()
            diag_idx = np.arange(3)
            P_c[diag_idx, diag_idx] = np.clip(
                np.diag(P_c), 0.0, float(odom_max_variance)
            )
            P_c = 0.5 * (P_c + P_c.T)  # enforce symmetry after clamp
            return Pi @ P_c @ Pi.T

        V1 = V1 + _clamp_and_project(P_odom1, Pi1)
        V2 = V2 + _clamp_and_project(P_odom2, Pi2)

    # ── Step 2 (cont.): Build weights W_i = inv(V_i) ─────────────────────────
    # V_i is SPD because the isotropic visual term guarantees full rank.
    try:
        W1 = np.linalg.inv(V1)
        W2 = np.linalg.inv(V2)
    except np.linalg.LinAlgError:
        # Pathological: fall back to identity weighting
        W1 = I3
        W2 = I3
        if logger is not None:
            logger.warn(
                '[triangulate_wls] V_i singular during inversion. '
                'Falling back to identity weights.',
                throttle_duration_sec=5.0,
            )

    # ── Step 2 (cont.): Fisher Information Matrix and WLS solution ───────────
    # Lambda = Pi1^T W1 Pi1 + Pi2^T W2 Pi2  (equivalent to A^T W A)
    # Computed block-by-block to avoid allocating 6x6 matrices.
    Pi1W1Pi1 = Pi1.T @ W1 @ Pi1   # (3x3)
    Pi2W2Pi2 = Pi2.T @ W2 @ Pi2   # (3x3)
    Lambda = Pi1W1Pi1 + Pi2W2Pi2   # Fisher Information Matrix (3x3)

    rhs = Pi1W1Pi1 @ origin1 + Pi2W2Pi2 @ origin2   # (3,)

    # ── Step 3: Output covariance (CRLB) and MLE position ────────────────────
    try:
        R_out = np.linalg.inv(Lambda)
    except np.linalg.LinAlgError:
        # Should never occur given the observability check, but be defensive.
        R_out = I3 * 1e6
        if logger is not None:
            logger.warn(
                '[triangulate_wls] Lambda singular. Returning high-uncertainty R.',
                throttle_duration_sec=5.0,
            )

    p_hat = R_out @ rhs

    # ── CRLB Normalisation (controlled by normalize_crlb flag) ───────────────
    #
    # Background:
    #   R_out = Lambda^{-1} is the exact Cramér-Rao Lower Bound (CRLB) for this
    #   geometry.  Its absolute magnitude is O(var_px * d²), which grows
    #   quadratically with target distance and can reach ~1000–3000 m² at 15 m.
    #
    # Option A — normalize_crlb = True  (legacy / pragmatic, DEFAULT)
    #   Preserve the ANISOTROPIC SHAPE of the CRLB (eigenvalue ratios encode
    #   which Cartesian directions are well-observed) but rescale the magnitude
    #   so that trace(R) = 3 * var_px.  This matches the isotropic scalar
    #   `R = var_px * I` in the limit of equal weights and 90° geometry, letting
    #   kf_measurement_noise act as the global trust knob exactly as before.
    #   → Backward compatible: existing kf_measurement_noise and kf_process_noise_*
    #     parameters remain valid without re-tuning.
    #   → Academic note: this is a "normalised-CRLB shape" approach (Ristic et al.,
    #     2004, "Beyond the Kalman Filter"), defensible but not SOTA.
    #
    # Option B — normalize_crlb = False  (physically rigorous, SOTA)
    #   Return R_out = Lambda^{-1} unchanged.  The Kalman Filter receives the
    #   true measurement uncertainty; the Kalman Gain automatically decreases at
    #   large distances (correct physical behaviour) and increases when geometry
    #   is favourable.
    #   → Requires the process-noise matrices Q_cv and Q_ca to also be on the
    #     correct physical scale.  CMA-ES must be re-run with wider Q bounds and
    # else: R_out = Lambda^{-1} — raw CRLB, physically rigorous.

    # Enforce symmetry (floating-point rounding)
    R_out = 0.5 * (R_out + R_out.T)

    return p_hat, R_out



# A bearing carries no range information. 10x the range is "I know nothing along this ray"
# while staying far from the numerical noise floor of the information fusion below.
_RANGE_IGNORANCE_FACTOR = 10.0


def compute_triangulation_measurement_covariance(
    midpoint: np.ndarray,
    origin1: np.ndarray,
    dir1: np.ndarray,
    origin2: np.ndarray,
    dir2: np.ndarray,
    kf_measurement_noise: float,
    P_odom1: Optional[np.ndarray] = None,
    P_odom2: Optional[np.ndarray] = None,
    min_bearing_distance: float = 0.5,
    logger=None,
) -> np.ndarray:
    """Fisher-information triangulation covariance R (3×3) for a bearing-only pair.

    Maps angular measurement uncertainty into an anisotropic Cartesian ellipsoid.
    When observer odometry covariances ``P_odom1`` / ``P_odom2`` are provided,
    the observer position uncertainty is propagated into the bearing measurement
    noise via the first-order Jacobian (error-propagation law):

        R_total_i = R_pixel_i + J_i @ P_odom_i @ J_iᵀ

    where ``J_i = ∂b_i/∂p_obs_i`` is computed by
    :func:`compute_bearing_jacobian_wrt_observer`.

    This follows the standard linearised covariance propagation described in:

        Bar-Shalom, Li & Kirubarajan (2001), *Estimation with Applications to
        Tracking and Navigation*, Ch. 2 — "Error Propagation".

    and is directly related to the Fisher-Information weighting advocated for
    cooperative bearing-only tracking in:

        Olfati-Saber (2006), "Swarm Stability and Optimization", IEEE CDC;
        Zhao et al. (2013), "Bearing-only formation control", IEEE TAC.

    Args:
        midpoint:             Triangulated 3-D midpoint (best estimate of target).
        origin1/2:            3-D origins of the two bearing rays (camera centres).
        dir1/2:               Unit bearing directions.
        kf_measurement_noise: Bearing angular variance (rad²), e.g. 0.0019 = sigma 2.5 deg.
        P_odom1/2:            Optional 3×3 position covariance of each observer.
                              Pass ``None`` to fall back to pixel-only noise.
        min_bearing_distance: Clamp passed to Jacobian helper (m).
        logger:               Optional rclpy logger for validation warnings.

    Returns:
        R (3×3 float64 ndarray): Symmetric, positive-definite measurement
        covariance for the IMM update step.
    """
    d1 = float(max(np.linalg.norm(midpoint - origin1), 0.1))
    d2 = float(max(np.linalg.norm(midpoint - origin2), 0.1))
    # kf_measurement_noise is an ANGULAR variance (rad²), same convention as
    # compute_single_ray_measurement_covariance. Dividing it by the current range before
    # multiplying by d_i² cancelled out, pinning the lateral sigma to a fixed ~4 cm at any
    # range: the filter was told a 49 m shot was centimetre-accurate while it sat 25 m off.
    angular_var = max(float(kf_measurement_noise), 1e-8)
    var1 = (d1 ** 2) * angular_var
    var2 = (d2 ** 2) * angular_var

    proj1 = np.eye(3, dtype=np.float64) - np.outer(dir1, dir1)
    proj2 = np.eye(3, dtype=np.float64) - np.outer(dir2, dir2)

    # Base pixel-only information matrices. proj_i alone is rank-2 (zero variance ALONG the
    # ray), so the eps below used to invert to 1e6 of fictitious range information — a single
    # bearing constrains nothing along its own ray. Seed that axis with a deliberately huge
    # variance instead, so the fused R keeps the real bearing-only ellipsoid: tight across
    # both rays, elongated along the bisector when the parallax is weak.
    var_along1 = (_RANGE_IGNORANCE_FACTOR * d1) ** 2
    var_along2 = (_RANGE_IGNORANCE_FACTOR * d2) ** 2
    R_pix1 = proj1 * var1 + np.outer(dir1, dir1) * var_along1  # (3×3) — ray 1
    R_pix2 = proj2 * var2 + np.outer(dir2, dir2) * var_along2  # (3×3) — ray 2

    # Propagate observer position uncertainty (if available)
    # R_total_i = R_pixel_i + J_i @ P_odom_i @ J_iᵀ
    if P_odom1 is not None:
        J1 = compute_bearing_jacobian_wrt_observer(
            origin1, midpoint, min_distance=min_bearing_distance
        )
        odom_contrib1 = (d1 ** 2) * (J1 @ P_odom1 @ J1.T)
        R_pix1 = R_pix1 + odom_contrib1

    if P_odom2 is not None:
        J2 = compute_bearing_jacobian_wrt_observer(
            origin2, midpoint, min_distance=min_bearing_distance
        )
        odom_contrib2 = (d2 ** 2) * (J2 @ P_odom2 @ J2.T)
        R_pix2 = R_pix2 + odom_contrib2

    # Fisher information fusion: I_total = I_1 + I_2
    # (inverting each R_i gives a proper information weighting)
    eps = 1e-6 * np.eye(3, dtype=np.float64)
    try:
        info1 = np.linalg.inv(R_pix1 + eps)
    except np.linalg.LinAlgError:
        info1 = np.eye(3, dtype=np.float64) / max(var1, 1e-6)
    try:
        info2 = np.linalg.inv(R_pix2 + eps)
    except np.linalg.LinAlgError:
        info2 = np.eye(3, dtype=np.float64) / max(var2, 1e-6)

    try:
        R_out = np.linalg.inv(info1 + info2)
    except np.linalg.LinAlgError:
        R_out = np.eye(3, dtype=np.float64) * (var1 + var2) * 10.0

    return _validate_covariance_matrix(R_out, 'R_triangulation', logger=logger)


def compute_single_ray_measurement_covariance(
    ray_dir_world: np.ndarray,
    depth_m: float,
    depth_sigma_m: float,
    kf_measurement_noise: float,
) -> np.ndarray:
    """Fisher-information covariance R (3x3) for a single RGB-D ray.

    Maps radial depth uncertainty and lateral angular uncertainty into an
    anisotropic Cartesian ellipsoid.
    """
    var_radial = float(depth_sigma_m ** 2)
    var_lateral = (float(depth_m) ** 2) * float(kf_measurement_noise)
    
    proj_radial = np.outer(ray_dir_world, ray_dir_world)
    proj_lateral = np.eye(3, dtype=np.float64) - proj_radial
    
    R_single = var_radial * proj_radial + var_lateral * proj_lateral
    assert np.linalg.matrix_rank(R_single) == 3, "Degenerate single-ray covariance"
    return R_single


@dataclass(frozen=True)
class PredictionGuidedPairScore:
    """Cost breakdown for one feasible observer ray pair."""

    cost: float
    time_term: float
    geom_term: float
    innov_term: float
    midpoint: np.ndarray
    ray_gap_m: float


def score_prediction_guided_pair(
    origin1: np.ndarray,
    dir1: np.ndarray,
    origin2: np.ndarray,
    dir2: np.ndarray,
    pair_diff_ns: int,
    max_pair_age_ns: int,
    max_triangulation_error_m: float,
    triangulation_min_baseline_m: float,
    kf_measurement_noise: float,
    x_pred: np.ndarray,
    P_pred_pos: Optional[np.ndarray],
    weight_time: float,
    weight_geom: float,
    weight_innov: float,
    pairing_mahalanobis_chi2: float,
    imm_initialized: bool,
) -> Optional[PredictionGuidedPairScore]:
    """Score one ray pair for prediction-guided association (lower is better).

    Hard-rejects infeasible geometry before scoring. Innovation term uses
    S = P_pred_pos + R_3d and is disabled until the IMM has been initialized.
    """
    if pair_diff_ns > max_pair_age_ns:
        return None

    baseline = float(np.linalg.norm(origin2 - origin1))
    if baseline < triangulation_min_baseline_m:
        return None

    midpoint, ray_gap_m = triangulate_midpoint(origin1, dir1, origin2, dir2)
    if ray_gap_m > max_triangulation_error_m:
        return None

    dt_max = max(float(max_pair_age_ns), 1.0)
    e_max = max(float(max_triangulation_error_m), 1e-6)
    chi2_norm = max(float(pairing_mahalanobis_chi2), 1e-6)

    time_term = float(pair_diff_ns) / dt_max
    geom_term = float(ray_gap_m) / e_max

    innov_term = 0.0
    if imm_initialized and P_pred_pos is not None:
        R = compute_triangulation_measurement_covariance(
            midpoint,
            origin1,
            dir1,
            origin2,
            dir2,
            kf_measurement_noise,
        )
        S = P_pred_pos + R
        innovation = midpoint - x_pred
        try:
            S_inv = np.linalg.inv(S)
            d2 = float(innovation.T @ S_inv @ innovation)
        except np.linalg.LinAlgError:
            return None
        innov_term = d2 / chi2_norm

    cost = (
        float(weight_time) * time_term
        + float(weight_geom) * geom_term
        + float(weight_innov) * innov_term
    )
    return PredictionGuidedPairScore(
        cost=cost,
        time_term=time_term,
        geom_term=geom_term,
        innov_term=innov_term,
        midpoint=midpoint,
        ray_gap_m=ray_gap_m,
    )


def select_timestamp_only_pair(
    observations1: List['SpatialObservation'],
    observations2: List['SpatialObservation'],
    max_pair_age_ns: int,
) -> Optional[Tuple['SpatialObservation', 'SpatialObservation', int]]:
    """Legacy pairing: choose the pair with minimum absolute timestamp difference."""
    best_obs1 = None
    best_obs2 = None
    best_diff = max_pair_age_ns + 1

    for obs1 in observations1:
        for obs2 in observations2:
            diff = abs(obs1.stamp_ns - obs2.stamp_ns)
            if diff < best_diff:
                best_diff = diff
                best_obs1 = obs1
                best_obs2 = obs2

    if best_obs1 is None or best_obs2 is None or best_diff > max_pair_age_ns:
        return None
    return best_obs1, best_obs2, best_diff


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

    def correct_1d(self, sc: StateCov, z_scalar: float, h_vec: np.ndarray, r_scalar: float) -> StateCov:
        """Sequential 1D update along a specific measurement vector h_vec (1xN)."""
        y = z_scalar - float(h_vec @ sc.x)
        S = float(h_vec @ sc.P @ h_vec.T) + r_scalar
        K = (sc.P @ h_vec.T) / max(S, 1e-12)
        x_new = sc.x + K * y
        P_new = (np.eye(self.n) - np.outer(K, h_vec)) @ sc.P
        return StateCov(x=x_new, P=P_new)


def build_transition_matrix_ca(dt: float, tau: float = 0.5) -> np.ndarray:
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
    
    # Singer Model Acceleration Decay
    alpha = np.exp(-dt / tau)
    A[6, 6] = alpha
    A[7, 7] = alpha
    A[8, 8] = alpha
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
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool, float]:
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

        # ── Combined Innovation and Covariance (Mixture Reduction) ──
        # Evaluated using the predicted mode probabilities (c_bar)
        nu_comb = c_bar[0] * y_cv + c_bar[1] * y_ca
        S_comb = (c_bar[0] * S_cv + c_bar[1] * S_ca) + \
                 c_bar[0] * np.outer(y_cv - nu_comb, y_cv - nu_comb) + \
                 c_bar[1] * np.outer(y_ca - nu_comb, y_ca - nu_comb)
        
        try:
            invS_comb = np.linalg.inv(S_comb)
            combined_nis = float(nu_comb.T @ invS_comb @ nu_comb)
        except np.linalg.LinAlgError:
            combined_nis = float('inf')

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
            return x_comb, P_comb, self.mu, accepted, combined_nis

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

        return x_comb, P_comb, self.mu, accepted, combined_nis

    def correct_1d(
        self,
        z_scalar: float,
        h_vec_cv: np.ndarray,
        h_vec_ca: np.ndarray,
        r_scalar: float,
        chi2_thresh: float = 9.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        """Apply a 1D scalar update (e.g., depth fusion) to both CV and CA filters."""
        assert self.sc_cv is not None
        assert self.sc_ca is not None

        # Calculate innovations and S (variance) for 1D
        y_cv = z_scalar - float(h_vec_cv @ self.sc_cv.x)
        S_cv = float(h_vec_cv @ self.sc_cv.P @ h_vec_cv.T) + r_scalar
        
        y_ca = z_scalar - float(h_vec_ca @ self.sc_ca.x)
        S_ca = float(h_vec_ca @ self.sc_ca.P @ h_vec_ca.T) + r_scalar

        # Mahalanobis gating
        d_cv = (y_cv * y_cv) / max(S_cv, 1e-12)
        d_ca = (y_ca * y_ca) / max(S_ca, 1e-12)

        accepted = True
        if d_cv > chi2_thresh and d_ca > chi2_thresh:
            accepted = False
            # Reject: keep current state and mu
            x_comb = self._state_cv_to_ca(self.sc_cv.x) * self.mu[0] + self.sc_ca.x * self.mu[1]
            dx0_comb = self._state_cv_to_ca(self.sc_cv.x) - x_comb
            dx1_comb = self.sc_ca.x - x_comb
            P_comb = (
                self.mu[0] * (self._cov_cv_to_ca(self.sc_cv.P) + np.outer(dx0_comb, dx0_comb))
                + self.mu[1] * (self.sc_ca.P + np.outer(dx1_comb, dx1_comb))
            )
            return x_comb, P_comb, self.mu, accepted

        # Accept: Update modes
        sc_upd_cv = self.kf_cv.correct_1d(self.sc_cv, z_scalar, h_vec_cv, r_scalar)
        sc_upd_ca = self.kf_ca.correct_1d(self.sc_ca, z_scalar, h_vec_ca, r_scalar)

        # Update probabilities
        norm_cv = 1.0 / np.sqrt(2 * np.pi * max(S_cv, 1e-12))
        likelihood_cv = norm_cv * np.exp(max(-50.0, min(50.0, -0.5 * d_cv)))

        norm_ca = 1.0 / np.sqrt(2 * np.pi * max(S_ca, 1e-12))
        likelihood_ca = norm_ca * np.exp(max(-50.0, min(50.0, -0.5 * d_ca)))

        Lambda = np.array([likelihood_cv, likelihood_ca])
        self.mu = Lambda * self.mu
        self.mu /= np.sum(self.mu) + 1e-12

        self.sc_cv = sc_upd_cv
        self.sc_ca = sc_upd_ca

        # Re-mix to output combined state
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
    depth_valid: bool = False
    depth_m: float = 0.0
    depth_sigma_m: float = 0.0


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
        # Anisotropic bearing noise: OFF by default, so the isotropic
        # kf_measurement_noise path stays bit-for-bit unchanged until enabled.
        _default_anisotropic_bearing_noise = False
        # TOTAL per-axis budget, attitude and detector added in quadrature:
        #   az = sqrt(1.04^2 + 0.81^2)  yaw from RTK baseline (+) detector jitter
        #   el = sqrt(2.50^2 + 0.81^2)  tilt from boot level ref (+) detector jitter
        _default_bearing_sigma_az_deg = 1.32
        _default_bearing_sigma_el_deg = 2.63
        _default_adaptive_cov_max_multiplier = 50.0
        _default_initial_covariance = 10.0
        _default_mahalanobis_threshold = 9.0
        _default_max_pair_age_sec = 0.25
        _default_observation_buffer_sec = 1.0
        _default_triangulation_min_baseline_m = 0.25
        _default_max_triangulation_error_m = 2.0
        _default_processing_rate_hz = 100.0
        _default_publish_evaluation_packets = False
        _default_pairing_prediction_enabled = False
        _default_pairing_weight_time = 1.0
        _default_pairing_weight_geom = 2.0
        _default_pairing_weight_innov = 3.0
        _default_pairing_mahalanobis_chi2 = 9.0
        _default_pairing_log_best_cost = False

        # Depth fusion defaults (for multi-modal triangulation)
        _default_df_use_range_constraint = True
        _default_df_range_chi2 = 9.0

        _default_single_obs_max_age_sec = 0.15
        _default_single_obs_min_confidence = 0.60
        _default_coast_after_sec = 0.15

        # Observer odometry uncertainty propagation defaults
        # Reference: Dogancay (2005/2015), Xu et al. (2017), Bishop et al. (2010).
        _default_use_odom_uncertainty = True
        _default_odom_fallback_variance = 0.01   # m²: conservative fallback when P_odom not published
        _default_min_bearing_distance = 0.5      # m:  kept for legacy; WLS bootstrap handles near-range
        _default_odom_max_age_sec = 0.5          # s:  stale odometry → fall back to pixel-only
        _default_distance_dependent_noise = True  # WLS: scale visual sigma by d^2
        _default_odom_max_variance = 1.0          # m²: diagonal clamp on P_odom (protects against sim baro Z~1286)
        _default_shared_heading = False           # observer1 adopts observer2's yaw (bench only)
        _bag_profile_name = 'none'                # which bag_profiles block was applied
        _bearing_bias_cfg = {}                    # per-role constant (az, el) ray bias, degrees
        _target_gt_z_offset_m = 0.0               # m: residual z the utm datum hop leaves on the GT

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
                        # Per-flight survey constants live under bag_profiles; bag_profile
                        # picks one and "none" picks nothing, so a new bag replays raw. The
                        # BAG_PROFILE env var overrides for a single run without editing yaml.
                        _bag_profile_name = os.environ.get(
                            'BAG_PROFILE') or _cfg.get('bag_profile') or 'none'
                        _profiles = _cfg.get('bag_profiles') or {}
                        if _bag_profile_name != 'none' and _bag_profile_name not in _profiles:
                            raise SystemExit(
                                f'bag_profile "{_bag_profile_name}" is not in {_config_path}. '
                                f'Known profiles: {sorted(_profiles) + ["none"]}')
                        _profile = _profiles.get(_bag_profile_name) or {}
                        _bearing_bias_cfg = _profile.get('bearing_bias') or {}
                        _target_gt_z_offset_m = float(_profile.get('target_gt_z_offset_m') or 0.0)
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
                            _abn = _ekf.get('anisotropic_bearing_noise', {}) or {}
                            _default_anisotropic_bearing_noise = bool(
                                _abn.get('enabled', _default_anisotropic_bearing_noise)
                            )
                            _default_bearing_sigma_az_deg = float(
                                _abn.get('sigma_az_deg', _default_bearing_sigma_az_deg)
                            )
                            _default_bearing_sigma_el_deg = float(
                                _abn.get('sigma_el_deg', _default_bearing_sigma_el_deg)
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
                        if 'triangulation' in _cfg:
                            _tri = _cfg['triangulation']
                            _default_max_pair_age_sec = float(
                                _tri.get('max_pair_age_sec', _default_max_pair_age_sec)
                            )
                            _default_observation_buffer_sec = float(
                                _tri.get('observation_buffer_sec', _default_observation_buffer_sec)
                            )
                            _default_triangulation_min_baseline_m = float(
                                _tri.get('triangulation_min_baseline_m', _default_triangulation_min_baseline_m)
                            )
                            _default_max_triangulation_error_m = float(
                                _tri.get('max_triangulation_error_m', _default_max_triangulation_error_m)
                            )
                            _default_processing_rate_hz = float(
                                _tri.get('processing_rate_hz', _default_processing_rate_hz)
                            )
                            _default_publish_evaluation_packets = bool(
                                _tri.get('publish_evaluation_packets', _default_publish_evaluation_packets)
                            )
                            _default_pairing_prediction_enabled = bool(
                                _tri.get('pairing_prediction_enabled', _default_pairing_prediction_enabled)
                            )
                            _default_pairing_weight_time = float(
                                _tri.get('pairing_weight_time', _default_pairing_weight_time)
                            )
                            _default_pairing_weight_geom = float(
                                _tri.get('pairing_weight_geom', _default_pairing_weight_geom)
                            )
                            _default_pairing_weight_innov = float(
                                _tri.get('pairing_weight_innov', _default_pairing_weight_innov)
                            )
                            _default_pairing_mahalanobis_chi2 = float(
                                _tri.get('pairing_mahalanobis_chi2', _default_pairing_mahalanobis_chi2)
                            )
                            _default_pairing_log_best_cost = bool(
                                _tri.get('pairing_log_best_cost', _default_pairing_log_best_cost)
                            )
                            _default_single_obs_max_age_sec = float(
                                _tri.get('single_obs_max_age_sec', _default_single_obs_max_age_sec)
                            )
                            _default_single_obs_min_confidence = float(
                                _tri.get('single_obs_min_confidence', _default_single_obs_min_confidence)
                            )
                            _default_coast_after_sec = float(
                                _tri.get('coast_after_sec', _default_coast_after_sec)
                            )
                        if 'depth_fusion' in _cfg:
                            _df = _cfg['depth_fusion']
                            _default_df_use_range_constraint = bool(
                                _df.get('fusion_use_range_constraint', _default_df_use_range_constraint)
                            )
                            _default_df_range_chi2 = float(
                                _df.get('fusion_range_chi2', _default_df_range_chi2)
                            )
                        _default_shared_heading = bool(
                            _cfg.get('shared_heading', _default_shared_heading))
                        if 'observer_odom_uncertainty' in _cfg:
                            _oou = _cfg['observer_odom_uncertainty']
                            _default_use_odom_uncertainty = bool(
                                _oou.get('use_odom_uncertainty', _default_use_odom_uncertainty)
                            )
                            _default_odom_fallback_variance = float(
                                _oou.get('odom_fallback_variance', _default_odom_fallback_variance)
                            )
                            _default_min_bearing_distance = float(
                                _oou.get('min_bearing_distance', _default_min_bearing_distance)
                            )
                            _default_odom_max_age_sec = float(
                                _oou.get('odom_max_age_sec', _default_odom_max_age_sec)
                            )
                            _default_distance_dependent_noise = bool(
                                _oou.get('distance_dependent_noise', _default_distance_dependent_noise)
                            )
                            _default_odom_max_variance = float(
                                _oou.get('odom_max_variance', _default_odom_max_variance)
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
        self.declare_parameter('gt_aligned_topic', '/target/ground_truth_aligned')
        self.declare_parameter('target_frame_id', 'world')
        self.declare_parameter('max_pair_age_sec', _default_max_pair_age_sec)
        self.declare_parameter('observation_buffer_sec', _default_observation_buffer_sec)
        self.declare_parameter('triangulation_min_baseline_m', _default_triangulation_min_baseline_m)
        self.declare_parameter('max_triangulation_error_m', _default_max_triangulation_error_m)
        self.declare_parameter('max_target_gt_age_sec', 0.75)
        # Field scale, not a guess at "big": the observers sit ~9 m apart on one field and
        # their world origins agree to ~1 cm in XY and a few metres in Z, so a 120 m datum
        # (seen on v6 while GNSS was still converging) is unphysical by three orders of
        # magnitude. Raise this only if the drones genuinely boot far apart.
        self.declare_parameter('datum_max_offset_m', 50.0)
        self.declare_parameter('datum_settle_tol_m', 0.05)
        # Each refresh costs two tf lookups that can block for tf_timeout_sec, and they sit
        # on the 76 Hz GT path: refreshing every 0.5 s starved the pipeline to a fifth of
        # its estimate rate on v6. 2 s keeps throughput while still filling a 15-sample
        # median window inside the first 30 s, which is where the datum is worst.
        self.declare_parameter('datum_refresh_sec', 2.0)
        self.declare_parameter('datum_window', 15)
        self.declare_parameter('processing_rate_hz', _default_processing_rate_hz)
        self.declare_parameter('pairing_prediction_enabled', _default_pairing_prediction_enabled)
        self.declare_parameter('pairing_weight_time', _default_pairing_weight_time)
        self.declare_parameter('pairing_weight_geom', _default_pairing_weight_geom)
        self.declare_parameter('pairing_weight_innov', _default_pairing_weight_innov)
        self.declare_parameter('pairing_mahalanobis_chi2', _default_pairing_mahalanobis_chi2)
        self.declare_parameter('pairing_log_best_cost', _default_pairing_log_best_cost)
        self.declare_parameter('single_obs_max_age_sec', _default_single_obs_max_age_sec)
        self.declare_parameter('single_obs_min_confidence', _default_single_obs_min_confidence)
        self.declare_parameter('coast_after_sec', _default_coast_after_sec)
        self.declare_parameter('df_use_range_constraint', _default_df_use_range_constraint)
        self.declare_parameter('df_range_chi2', _default_df_range_chi2)
        self.declare_parameter('shared_heading', _default_shared_heading)
        self.declare_parameter('use_odom_uncertainty', _default_use_odom_uncertainty)
        self.declare_parameter('odom_fallback_variance', _default_odom_fallback_variance)
        self.declare_parameter('min_bearing_distance', _default_min_bearing_distance)
        self.declare_parameter('odom_max_age_sec', _default_odom_max_age_sec)
        self.declare_parameter('distance_dependent_noise', _default_distance_dependent_noise)
        self.declare_parameter('odom_max_variance', _default_odom_max_variance)
        self.declare_parameter('tf_timeout_sec', 0.05)
        self.declare_parameter('world_frame_suffix', _world_frame_suffix)
        self.declare_parameter('kf_process_noise_acc', _default_process_noise_acc)
        self.declare_parameter('kf_process_noise_vel', _default_process_noise_vel)
        self.declare_parameter('imm_transition_prob', _default_imm_transition_prob)  # legacy, unused
        self.declare_parameter('imm_p_stay_cv', 0.80)       # P(CV→CV): high = strong straight-flight bias
        self.declare_parameter('imm_p_return_to_cv', 0.30)  # P(CA→CV): high = fast decay of CA after maneuver
        self.declare_parameter('kf_measurement_noise', _default_measurement_noise)
        # normalize_crlb: True = rescale CRLB trace to 3*kf_measurement_noise (legacy, backward-compat).
        # False = pass raw Lambda^{-1} to IMM (SOTA; requires CMA-ES re-run with wider Q bounds).
        self.declare_parameter('normalize_crlb', True)
        # Anisotropic bearing noise (ekf.anisotropic_bearing_noise in runtime_stack.yaml).
        # enabled=False  -> visual term is var_px * d^2 * I_3          (isotropic cone)
        # enabled=True   -> visual term has separate az/el variances   (elliptical cone)
        # Sigmas are given in DEGREES in the YAML and converted to rad^2 below.
        self.declare_parameter('anisotropic_bearing_noise', _default_anisotropic_bearing_noise)
        self.declare_parameter('bearing_sigma_az_deg', _default_bearing_sigma_az_deg)
        self.declare_parameter('bearing_sigma_el_deg', _default_bearing_sigma_el_deg)
        self.declare_parameter('kf_adaptive_cov_max_multiplier', _default_adaptive_cov_max_multiplier)
        self.declare_parameter('kf_initial_covariance', _default_initial_covariance)
        self.declare_parameter('kf_mahalanobis_threshold', _default_mahalanobis_threshold)
        self.declare_parameter('kf_prediction_horizon_sec', 1.0)
        self.declare_parameter('kf_prediction_steps', 25)
        self.declare_parameter('publish_evaluation_packets', _default_publish_evaluation_packets)
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
        self.gt_aligned_topic = str(self.get_parameter('gt_aligned_topic').value)
        self.target_frame_id = str(self.get_parameter('target_frame_id').value)
        self.max_pair_age_ns = int(float(self.get_parameter('max_pair_age_sec').value) * 1_000_000_000.0)
        self.observation_buffer_ns = int(float(self.get_parameter('observation_buffer_sec').value) * 1_000_000_000.0)
        self.triangulation_min_baseline_m = float(self.get_parameter('triangulation_min_baseline_m').value)
        self.max_triangulation_error_m = float(self.get_parameter('max_triangulation_error_m').value)
        self.max_target_gt_age_ns = int(float(self.get_parameter('max_target_gt_age_sec').value) * 1_000_000_000.0)
        self.datum_max_offset_m = float(self.get_parameter('datum_max_offset_m').value)
        self.datum_settle_tol_m = float(self.get_parameter('datum_settle_tol_m').value)
        self.datum_refresh_ns = int(float(self.get_parameter('datum_refresh_sec').value) * 1e9)
        self.datum_window = int(self.get_parameter('datum_window').value)
        self.processing_rate_hz = max(0.5, float(self.get_parameter('processing_rate_hz').value))
        self.tf_timeout_sec = float(self.get_parameter('tf_timeout_sec').value)
        self.world_frame_suffix = str(self.get_parameter('world_frame_suffix').value)
        self.kf_process_noise_acc = float(self.get_parameter('kf_process_noise_acc').value)
        self.kf_process_noise_vel = float(self.get_parameter('kf_process_noise_vel').value)
        self.imm_transition_prob = float(self.get_parameter('imm_transition_prob').value)  # legacy
        self.imm_p_stay_cv = float(self.get_parameter('imm_p_stay_cv').value)
        self.imm_p_return_to_cv = float(self.get_parameter('imm_p_return_to_cv').value)
        self.kf_measurement_noise = float(self.get_parameter('kf_measurement_noise').value)
        # normalize_crlb: controls whether the raw CRLB (inv(Lambda)) is rescaled to
        # trace=3*kf_measurement_noise (True, legacy) or passed unchanged to the IMM (False, SOTA).
        # Set to False only after re-running CMA-ES with wider Q bounds and without
        # kf_measurement_noise in the optimisation vector.
        self.normalize_crlb = bool(self.get_parameter('normalize_crlb').value)

        # Anisotropic bearing noise: convert sigma [deg] -> variance [rad^2] once here,
        # so triangulate_wls() only ever deals with rad^2 like var_px does.
        self.anisotropic_bearing_noise = bool(
            self.get_parameter('anisotropic_bearing_noise').value
        )
        self.bearing_sigma_az_deg = float(self.get_parameter('bearing_sigma_az_deg').value)
        self.bearing_sigma_el_deg = float(self.get_parameter('bearing_sigma_el_deg').value)
        self.bearing_var_az = float(np.deg2rad(self.bearing_sigma_az_deg)) ** 2
        self.bearing_var_el = float(np.deg2rad(self.bearing_sigma_el_deg)) ** 2
        self.kf_adaptive_cov_max_multiplier = float(self.get_parameter('kf_adaptive_cov_max_multiplier').value)
        self.kf_initial_covariance = float(self.get_parameter('kf_initial_covariance').value)
        self.kf_mahalanobis_threshold = float(self.get_parameter('kf_mahalanobis_threshold').value)
        self.kf_prediction_horizon_sec = float(self.get_parameter('kf_prediction_horizon_sec').value)
        self.kf_prediction_steps = max(5, int(self.get_parameter('kf_prediction_steps').value))
        self.pairing_prediction_enabled = bool(self.get_parameter('pairing_prediction_enabled').value)
        self.pairing_weight_time = float(self.get_parameter('pairing_weight_time').value)
        self.pairing_weight_geom = float(self.get_parameter('pairing_weight_geom').value)
        self.pairing_weight_innov = float(self.get_parameter('pairing_weight_innov').value)
        self.pairing_mahalanobis_chi2 = float(self.get_parameter('pairing_mahalanobis_chi2').value)
        self.pairing_log_best_cost = bool(self.get_parameter('pairing_log_best_cost').value)
        self.single_obs_max_age_ns = int(float(self.get_parameter('single_obs_max_age_sec').value) * 1_000_000_000.0)
        self.single_obs_min_confidence = float(self.get_parameter('single_obs_min_confidence').value)
        self.coast_after_sec = float(self.get_parameter('coast_after_sec').value)
        self.df_use_range_constraint = bool(self.get_parameter('df_use_range_constraint').value)
        self.df_range_chi2 = float(self.get_parameter('df_range_chi2').value)
        self.shared_heading = bool(self.get_parameter('shared_heading').value)
        self.use_odom_uncertainty = bool(self.get_parameter('use_odom_uncertainty').value)
        self.odom_fallback_variance = float(self.get_parameter('odom_fallback_variance').value)
        self.min_bearing_distance = float(self.get_parameter('min_bearing_distance').value)
        self.odom_max_age_ns = int(float(self.get_parameter('odom_max_age_sec').value) * 1_000_000_000.0)
        self.distance_dependent_noise = bool(self.get_parameter('distance_dependent_noise').value)
        self.odom_max_variance = float(self.get_parameter('odom_max_variance').value)
        self.publish_evaluation_packets = bool(self.get_parameter('publish_evaluation_packets').value)
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

        # IMM Transition Probability Matrix (Markov Chain) — configurable from runtime_stack.yaml
        # [CV→CV, CV→CA]
        # [CA→CV, CA→CA]
        #
        # imm_p_stay_cv (default 0.80):
        #   Probability of remaining in CV (straight-flight) model.
        #   Lower = CA activates faster at braking onset.
        #   Too low (<0.60) = jittery acceleration, noisy straight flight.
        #
        # imm_p_return_to_cv (default 0.30):
        #   Probability of CA decaying back to CV after a maneuver.
        #   Higher = CA fades faster; lower = CA persists longer through turns.
        #   Too low (<0.15) = filter "locks on" to ghost acceleration after curves.
        p_stay_cv = self.imm_p_stay_cv
        p_return_to_cv = self.imm_p_return_to_cv
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
        # Wallclock instant at which the filter last consumed a measurement.
        # Distinct from imm_last_stamp_ns (image capture time) on purpose: the
        # two live on different clocks and must never be subtracted from each
        # other.  Only the timer guard uses this one.
        self._last_measurement_wall_ns: int = 0
        self.filtered_position = np.zeros(3)
        self.filtered_velocity = np.zeros(3)
        self.filtered_acceleration = np.zeros(3)
        self.combined_cov = np.eye(9)
        self.imm_probs = mu0.copy()
        self._imm_cov_trace_reset_thresh = 500.0

        self.target_gt_odom: Optional[Odometry] = None
        # Origin-frame datum offsets are survey constants; resolve once per source frame.
        # source_frame -> (rot, trans, resolved_at_stamp_ns); expires after datum_refresh_sec.
        self._datum_cache: Dict[str, Tuple[np.ndarray, np.ndarray, int]] = {}
        # Recent accepted lookups per source_frame, median-filtered into the datum estimate.
        self._datum_samples: Dict[str, Deque[np.ndarray]] = collections.defaultdict(
            lambda: collections.deque(maxlen=self.datum_window))
        # Per-observer (stamp, az_err, el_err) history, for fitting a constant heading bias.
        # Bounded: the fit is re-run every 50 samples and is O(n), so an unbounded list makes
        # a long flight pay for its own history. 20k samples is ~11 min at 30 Hz per observer,
        # past any fit the drift regressor can still call constant.
        self._bias_samples: Dict[str, Deque] = collections.defaultdict(
            lambda: collections.deque(maxlen=20000))
        self._bias_sample_count = 0
        # Measured constant bearing bias per observer, keyed by uav name (see [BIASFIT] log).
        self.bearing_bias: Dict[str, Tuple[float, float]] = {}
        self.get_logger().info(f'Bag profile: {_bag_profile_name}')
        for _role, _uav in (('observer1', self.observer1_name), ('observer2', self.observer2_name)):
            _b = _bearing_bias_cfg.get(_role) or {}
            _az, _el = float(_b.get('az_deg', 0.0)), float(_b.get('el_deg', 0.0))
            if _az or _el:
                self.bearing_bias[_uav] = (_az, _el)
                self.get_logger().info(
                    f'Bearing bias correction for {_uav}: az={_az:+.2f} deg el={_el:+.2f} deg')
        if self.shared_heading:
            self.get_logger().warn(
                f'SHARED HEADING ACTIVE: {self.observer1_name} rays are rotated to use '
                f'{self.observer2_name} heading instead of their own. This is valid ONLY while '
                'both airframes are physically parallel (static bench experiments). In free '
                'flight it injects the real heading difference as a bearing error -- turn it off.')
        # Residual z the utm datum hop leaves on the target GT (see config for why it is on the GT).
        self.target_gt_z_offset_m = _target_gt_z_offset_m
        if self.target_gt_z_offset_m:
            self.get_logger().info(
                f'Target GT z datum offset: {self.target_gt_z_offset_m:+.2f} m')
        self.last_error_log_time = time.time()

        # Observer odometry for position uncertainty propagation into R.
        # Populated by _observer{1,2}_odom_callback; None when not yet received.
        self._observer1_odom: Optional[Odometry] = None
        self._observer1_odom_stamp_ns: int = 0
        self._observer2_odom: Optional[Odometry] = None
        self._observer2_odom_stamp_ns: int = 0

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.bridge = CvBridge() if self.publish_debug_annotated_image else None

        self.pose_pub = self.create_publisher(PoseStamped, self.pose_topic, 10)
        self.odometry_pub = self.create_publisher(Odometry, self.odometry_topic, 10)
        self.estimated_position_pub = self.create_publisher(Vector3Stamped, self.estimated_position_topic, 10)
        self.triangulation_raw_pub = self.create_publisher(PointStamped, self.triangulation_raw_topic, 10)
        self.error_vector_pub = self.create_publisher(Vector3Stamped, self.error_vector_topic, 10)
        self.gt_aligned_pub = self.create_publisher(Odometry, self.gt_aligned_topic, 20)
        self.triangulation_evaluation_pub = None
        if self.publish_evaluation_packets:
            self.triangulation_evaluation_pub = self.create_publisher(String, '/target/triangulation_evaluation', 10)
        self.predicted_trajectory_pub = self.create_publisher(Path, '/target/predicted_trajectory', 10)
        self.trajectory_markers_pub = self.create_publisher(MarkerArray, '/target/predicted_trajectory_markers', 10)
        self.position_markers_pub = self.create_publisher(MarkerArray, '/target/position_markers', 10)
        self.fusion_mode_pub = self.create_publisher(String, '/tracker/fusion_mode', 10)
        self.last_fusion_mode = "UNINITIALIZED"
        # Pairing/gating diagnostics: counters flushed to a one-line summary every 2 s.
        self._diag = collections.Counter()
        self._diag_last_log_ns = 0
        # Geometry probe: on failing cycles, measure the RAW ray-gap of time-feasible
        # candidate pairs and correlate it with dt, to separate a geometry/calibration
        # miss (gap large even at dt~0) from a time-sync miss (gap grows with dt).
        self._probe = {}
        self._probe_reset()
        self.observer1_debug_annotated_pub = None
        if self.publish_debug_annotated_image:
            self.observer1_debug_annotated_pub = self.create_publisher(
                CompressedImage,
                f'/{self.observer1_name}/triangulation_debug_annotated/compressed',
                10,
            )

        self.create_subscription(Odometry, self.target_gt_odom_topic, self.target_gt_odom_callback, 20)

        # ── Observer odometry subscribers (for pose uncertainty propagation) ────
        # Topic pattern mirrors the MRS UAV System standard: /{uav_name}/estimation_manager/odom_main
        # QoS BEST_EFFORT to match the drone's odometry publisher (avoids queue backpressure).
        # shared_heading needs both yaws every frame, so it forces the subscriptions on even
        # when the uncertainty path is off.
        if self.use_odom_uncertainty or self.shared_heading:
            _odom_qos = QoSProfile(
                depth=5,
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
            )
            self.create_subscription(
                Odometry,
                f'/{self.observer1_name}/estimation_manager/odom_main',
                self._observer1_odom_callback,
                _odom_qos,
            )
            self.create_subscription(
                Odometry,
                f'/{self.observer2_name}/estimation_manager/odom_main',
                self._observer2_odom_callback,
                _odom_qos,
            )
            _why = 'uncertainty propagation' if self.use_odom_uncertainty else 'shared_heading'
            self.get_logger().info(
                f'Observer odometry enabled for {_why}. '
                f'Subscribing to /{self.observer1_name}/estimation_manager/odom_main '
                f'and /{self.observer2_name}/estimation_manager/odom_main'
            )
        else:
            self.get_logger().info(
                'Observer odometry uncertainty DISABLED (use_odom_uncertainty=false). '
                'R computed from pixel noise only.'
            )

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
        pairing_mode = 'prediction-guided' if self.pairing_prediction_enabled else 'timestamp-only'
        self.get_logger().info(
            f'Ray pairing mode: {pairing_mode} '
            f'(w_t={self.pairing_weight_time:.2f}, w_g={self.pairing_weight_geom:.2f}, '
            f'w_i={self.pairing_weight_innov:.2f})'
        )
        self.get_logger().info(f'Observer 1 topic: {self.observer1_observation_topic}')
        self.get_logger().info(f'Observer 2 topic: {self.observer2_observation_topic}')
        self.get_logger().info(f'Observer 1 camera info topic: {self.observer1_camera_info_topic}')
        self.get_logger().info(f'Target frame: {self.target_frame_id}')
        self.get_logger().info(f'Observation buffer: {self.observation_buffer_ns / 1e9:.2f} s')
        if self.anisotropic_bearing_noise:
            self.get_logger().info(
                f'Bearing noise: ANISOTROPIC  '
                f'sigma_az={self.bearing_sigma_az_deg:.2f} deg '
                f'sigma_el={self.bearing_sigma_el_deg:.2f} deg  '
                f'(kf_measurement_noise={self.kf_measurement_noise:.2e} rad^2 unused for V_vis)'
            )
        else:
            self.get_logger().info(
                f'Bearing noise: ISOTROPIC  '
                f'kf_measurement_noise={self.kf_measurement_noise:.2e} rad^2 '
                f'(sigma={np.rad2deg(np.sqrt(self.kf_measurement_noise)):.2f} deg)'
            )
        if self.publish_debug_annotated_image:
            self.get_logger().info(
                f'Triangulation debug overlay topic: '
                f'/{self.observer1_name}/triangulation_debug_annotated/compressed'
            )

    def target_gt_odom_callback(self, msg: Odometry) -> None:
        self.target_gt_odom = msg

        # The raw GT sits in the target drone's own origin frame, several metres off in z from
        # the frame the estimate is published in. Every consumer that compared the two raw
        # topics (live monitor, evaluator) was therefore reporting the datum gap as tracking
        # error. Re-emit the aligned copy once here so they all share one frame.
        pos = self.gt_position_in_target_frame()
        if pos is None:
            return
        aligned = Odometry()
        aligned.header.stamp = msg.header.stamp
        aligned.header.frame_id = self.target_frame_id
        # Twist is expressed in child_frame_id (body), so a datum shift leaves it untouched.
        aligned.child_frame_id = msg.child_frame_id
        aligned.twist = msg.twist
        aligned.pose.pose.orientation = msg.pose.pose.orientation
        aligned.pose.pose.position.x = float(pos[0])
        aligned.pose.pose.position.y = float(pos[1])
        aligned.pose.pose.position.z = float(pos[2])
        self.gt_aligned_pub.publish(aligned)

    def _observer1_odom_callback(self, msg: Odometry) -> None:
        """Cache the latest odometry of observer 1 for position uncertainty propagation.

        Only invoked when ``use_odom_uncertainty=True``.  The message is stored
        alongside its ROS timestamp (nanoseconds) so that staleness can be checked
        at triangulation time without TF involvement.
        """
        self._observer1_odom = msg
        self._observer1_odom_stamp_ns = stamp_to_ns(msg.header.stamp)

    def _observer2_odom_callback(self, msg: Odometry) -> None:
        """Cache the latest odometry of observer 2 for position uncertainty propagation.

        Mirrors :meth:`_observer1_odom_callback`.
        """
        self._observer2_odom = msg
        self._observer2_odom_stamp_ns = stamp_to_ns(msg.header.stamp)

    def _get_observer_position_covariance(
        self,
        odom: Optional[Odometry],
        odom_stamp_ns: int,
        query_stamp_ns: int,
        observer_label: str,
    ) -> Optional[np.ndarray]:
        """Return the 3x3 position covariance for one observer, or None if unavailable.

        Freshness check: if the cached odometry is older than ``odom_max_age_ns``
        relative to the bearing measurement timestamp, we do NOT trust it and
        fall back to pixel-only noise (returns None).

        Args:
            odom:             Latest cached Odometry message (may be None).
            odom_stamp_ns:    ROS timestamp of the cached message (nanoseconds).
            query_stamp_ns:   Timestamp of the triangulation measurement (nanoseconds).
            observer_label:   Human-readable name for log throttling.

        Returns:
            3x3 position covariance ndarray, or None.
        """
        if odom is None:
            return None
        age_ns = abs(query_stamp_ns - odom_stamp_ns)
        if age_ns > self.odom_max_age_ns:
            self.get_logger().warn(
                f'\033[33m[odom_uncertainty] {observer_label} odometry stale '
                f'({age_ns / 1e6:.1f} ms > {self.odom_max_age_ns / 1e6:.1f} ms); '
                'falling back to pixel-only covariance.\033[0m',
                throttle_duration_sec=2.0,
            )
            return None
        return extract_position_covariance(
            odom,
            fallback_variance=self.odom_fallback_variance,
            logger=self.get_logger(),
            observer_label=observer_label
        )

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
            self._diag['dbg_no_obs'] += 1
            return

        debug_frame = self._find_closest_debug_image(source_name, local_obs.stamp_ns, self.debug_image_match_ns)
        if debug_frame is None:
            self._diag['dbg_no_frame'] += 1
            return

        matched_pair = self._find_closest_matched_pair(local_obs.stamp_ns, self.max_pair_age_ns)
        remote_stamp_ns = matched_pair.remote_obs.stamp_ns if matched_pair is not None else -1
        publish_key = (local_obs.stamp_ns, remote_stamp_ns)
        if self.last_debug_publish_key.get(source_name) == publish_key:
            self._diag['dbg_dedup'] += 1
            return

        if self.publish_debug_image(source_name, debug_frame, camera_info, local_obs, matched_pair):
            self.last_debug_publish_key[source_name] = publish_key
            self._diag['dbg_pub_mono' if matched_pair is None else 'dbg_pub_stereo'] += 1

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

        # transform_wc comes straight from tf and still carries this camera's compass bias, while
        # the rays have had theirs removed. Undo the correction for drawing only, or the local
        # (green) ray lands a full bias away from the target it was measured on.
        az_deg, el_deg = self._bearing_bias_for(source_name, local_obs.stamp_ns)
        if az_deg or el_deg:
            rot_wc = quat_to_rotmat(transform_wc.transform.rotation.x, transform_wc.transform.rotation.y,
                                    transform_wc.transform.rotation.z, transform_wc.transform.rotation.w)
            t_wc = np.array([transform_wc.transform.translation.x, transform_wc.transform.translation.y,
                             transform_wc.transform.translation.z], dtype=np.float64)
            cam_pos = -rot_wc.T @ t_wc
            rays_to_draw = [rebias_about(cam_pos, o, r, az_deg, el_deg) for o, r in rays_to_draw]

        # local_obs and remote_obs are expressed in semantically common world_origin coordinates.
        debug_img = project_ray_on_image(
            debug_frame.bgr_image,
            camera_info,
            transform_wc,
            rays_to_draw,
            color_list=colors,
            length=150.0,
        )
        self._draw_stereo_hud(debug_img, local_obs, matched_pair)

        debug_msg = CompressedImage()
        debug_msg.header = debug_frame.header
        debug_msg.format = 'jpeg'
        debug_msg.data = bytes(cv2.imencode('.jpg', debug_img, [cv2.IMWRITE_JPEG_QUALITY, 75])[1])
        debug_pub.publish(debug_msg)
        return True

    def _draw_stereo_hud(self, img, local_obs, matched_pair) -> None:
        """Say in the frame why the red ray is or is not there, and how much to trust the fix.

        A missing red ray looks exactly like a broken overlay, but on this bag it just means
        the second observer had no detection at that instant. Spell it out, and flag the
        range/baseline dilution that turns a ~2 deg bearing error into a huge depth error.
        """
        if matched_pair is None:
            line, color = 'MONO: no {} observation'.format(self.observer2_name), (0, 165, 255)
        else:
            local_a = matched_pair.aligned_local_obs
            remote_a = matched_pair.aligned_remote_obs
            dt_ms = abs(matched_pair.remote_obs.stamp_ns - local_obs.stamp_ns) / 1e6
            baseline = float(np.linalg.norm(remote_a.origin - local_a.origin))
            midpoint, gap = triangulate_midpoint(
                local_a.origin, local_a.ray, remote_a.origin, remote_a.ray)
            rng = float(np.linalg.norm(midpoint - local_a.origin))
            line = 'STEREO {} dt={:.0f}ms base={:.1f}m range={:.1f}m gap={:.2f}m'.format(
                self.observer2_name, dt_ms, baseline, rng, gap)
            color = (0, 255, 0)
            # Depth error ~ range^2 * bearing_sigma / baseline: past ~2 m of it the fix is a guess.
            # The ray gap does NOT catch this — a shared bearing bias slides the intersection
            # along the bisector without opening the gap at all.
            if baseline > 1e-3:
                depth_sigma = (rng ** 2) * np.sqrt(self.kf_measurement_noise) / baseline
                if depth_sigma > 2.0:
                    line += ' | WEAK GEOM sigma_depth={:.0f}m'.format(depth_sigma)
                    color = (0, 165, 255)
        cv2.putText(img, line, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, line, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

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

        stamp_ns, observation_id, confidence, origin, ray_raw, tf_delta_ns, tf_status, depth_valid, depth_m, depth_sigma_m = packet
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
            depth_valid=depth_valid,
            depth_m=depth_m,
            depth_sigma_m=depth_sigma_m,
        )
        obs = self._transform_observation_to_target_frame(obs)
        if obs is None:
            return
        obs = self._apply_bearing_bias(obs)
        self._collect_bias_sample(obs)

        self.debug_observations[source_name].append(obs)
        buffer.append(obs)
        self._diag['rx_' + source_name] += 1
        self._prune_stale_observations(stamp_ns)
        # Stereo triangulation is event-driven: attempt a match every time a new
        # observation arrives.  This is the correct approach because it minimises
        # latency and avoids the race-condition where the timer fires between the
        # arrival of obs1 and obs2 (leaving one buffer empty at match-time).
        if self.try_match_and_triangulate():
            self._publish_fusion_mode()
        self._try_publish_debug_image(source_name, stamp_ns)

    def _datum_rt(
        self,
        source_frame: str,
        stamp_ns: int,
        label: str = '',
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Rotation+translation mapping source_frame -> target frame, hopping through utm.

        Every drone roots its own tf tree at its own fcu, so a direct target<-source lookup
        fails. utm is the one grid all of them share, so we do two INTRA-tree lookups and
        compose. Measured on the v5 bag: the origin frames are exactly north-aligned (yaw 0)
        and share the XY datum to ~1 cm, but their Z datums differ by several metres
        (uav28 sits 5.39 m below uav81), so skipping this hop is a silent altitude bias.
        """
        cached = self._datum_cache.get(source_frame)
        if cached is not None and abs(stamp_ns - cached[2]) < self.datum_refresh_ns:
            return cached[0], cached[1]

        source_drone = source_frame.split('/', 1)[0]
        source_utm = f'{source_drone}/utm_origin'
        target_utm = f'{self.observer1_name}/utm_origin'
        lookup_time = Time(seconds=stamp_ns // 1_000_000_000, nanoseconds=stamp_ns % 1_000_000_000)

        def _look(target_frame: str, src_frame: str):
            try:
                return self.tf_buffer.lookup_transform(
                    target_frame, src_frame, lookup_time, Duration(seconds=self.tf_timeout_sec))
            except Exception:
                return self.tf_buffer.lookup_transform(
                    target_frame, src_frame, Time(), Duration(seconds=self.tf_timeout_sec))

        try:
            tf_to_utm = _look(source_utm, source_frame)
            tf_from_utm = _look(self.target_frame_id, target_utm)
        except Exception as exc:
            self.get_logger().warn(
                f'Cannot datum-align {label or source_frame} {source_frame} -> '
                f'{self.target_frame_id} via utm: {exc}',
                throttle_duration_sec=2.0,
            )
            return None

        def _rt(tf_msg):
            q = tf_msg.transform.rotation
            t = tf_msg.transform.translation
            return (quat_to_rotmat(q.x, q.y, q.z, q.w),
                    np.array([t.x, t.y, t.z], dtype=np.float64))

        rot1, trans1 = _rt(tf_to_utm)
        rot2, trans2 = _rt(tf_from_utm)
        rot = rot2 @ rot1
        trans = rot2 @ trans1 + trans2

        # The utm_origin of a drone whose GNSS has not converged yet is published anyway, and
        # composing through it yields a datum kilometres off. On the v6 bag with start-offset 95
        # this latched uav28 GT at (+674,-2197,-1003) m, which silently turned every GT-relative
        # number (bearing residuals, BIASFIT, ANEES) into noise while the node logged no error.
        # Drones flying the same field share a datum to ~1 cm in XY and a few metres in Z, so
        # anything beyond datum_max_offset_m is a bad lookup: refuse it and retry next message
        # rather than cache it for the rest of the flight.
        offset_m = float(np.linalg.norm(trans))
        if offset_m > self.datum_max_offset_m:
            self.get_logger().warn(
                f'Rejecting datum {source_frame} -> {self.target_frame_id}: offset '
                f'({trans[0]:+.1f},{trans[1]:+.1f},{trans[2]:+.1f}) m is {offset_m:.0f} m > '
                f'datum_max_offset_m={self.datum_max_offset_m:.0f} m; utm_origin not converged.',
                throttle_duration_sec=5.0,
            )
            return None

        # This datum is NOT a one-shot constant to be latched. It is composed from the two
        # drones' utm_origin frames, which are GNSS/RTK anchored and CONVERGE during the
        # flight: early in the v6 bag the same lookup returns 6118 m, then 1509 m, then
        # 120 m, and only later the true sub-metre value. Caching the first success froze
        # whichever garbage happened to arrive first and silently displaced every
        # triangulation for the rest of the run -- 2506 m on one start offset, 1.4 m on
        # another (which is worse, because nothing flags it).
        #
        # Guarding the latch does not work: a wrong value can sit perfectly still for
        # seconds while GNSS is still converging, so "hold still then latch" latched the
        # 120 m one. The fix is to stop latching. Re-resolve on a timer and always use the
        # freshest value, so the datum tracks convergence and self-corrects instead of
        # freezing. The cache still spares the 76 Hz GT path a tf lookup per message; it
        # just expires.
        # Accepted lookups still scatter by ~1 m about the truth (measured on v6: uav8 came
        # back -0.085, -1.207, -0.285, -0.076 m in Z within seconds), so neither the first
        # nor the freshest single sample is the datum. Using the freshest made the raw
        # midpoint error WORSE than the old frozen latch -- freezing a lucky good value
        # beats tracking a noisy one, and the old code was simply a lottery on which
        # sample it froze. The datum is physically a survey constant, so treat the scatter
        # as noise and estimate it: component-wise median over a short window of accepted
        # samples. Median rather than mean so one metre-scale outlier cannot drag it.
        samples = self._datum_samples[source_frame]
        samples.append(trans)
        estimate = np.median(np.stack(samples), axis=0) if len(samples) > 1 else trans

        previous = self._datum_cache.get(source_frame)
        self._datum_cache[source_frame] = (rot, estimate, stamp_ns)
        if previous is None:
            self.get_logger().info(
                f'Datum alignment {source_frame} -> {self.target_frame_id}: '
                f'offset=({estimate[0]:+.3f},{estimate[1]:+.3f},{estimate[2]:+.3f}) m'
            )
        elif float(np.linalg.norm(estimate - previous[1])) > self.datum_settle_tol_m:
            prev_trans = previous[1]
            self.get_logger().info(
                f'Datum alignment {source_frame} -> {self.target_frame_id} moved: '
                f'({prev_trans[0]:+.3f},{prev_trans[1]:+.3f},{prev_trans[2]:+.3f}) -> '
                f'({estimate[0]:+.3f},{estimate[1]:+.3f},{estimate[2]:+.3f}) m '
                f'(n={len(samples)})',
                throttle_duration_sec=5.0,
            )
        return rot, estimate

    def _shared_heading_az_deg(self, stamp_ns: int) -> float:
        """Azimuth offset that makes observer1's ray behave as if built with observer2's yaw.

        A world ray is R_z(psi) * ray_body, so swapping psi_1 for psi_2 means rotating by
        (psi_2 - psi_1); derotate_bearing SUBTRACTS its argument, hence (psi_1 - psi_2).

        Returns 0.0 when the mode is off, or when either odometry is missing or stale -- that
        frame simply keeps its own compass rather than being dropped, which is the right call
        for a bench run where odom arrives at ~100 Hz and a gap is a transport hiccup.
        """
        if not self.shared_heading:
            return 0.0
        if self._observer1_odom is None or self._observer2_odom is None:
            self.get_logger().warn(
                'shared_heading: observer odometry not received yet, rays left uncorrected',
                throttle_duration_sec=5.0)
            return 0.0
        age_ns = max(abs(stamp_ns - self._observer1_odom_stamp_ns),
                     abs(stamp_ns - self._observer2_odom_stamp_ns))
        if age_ns > self.odom_max_age_ns:
            self.get_logger().warn(
                f'shared_heading: odometry stale by {age_ns / 1e9:.2f} s '
                f'(max {self.odom_max_age_ns / 1e9:.2f}), rays left uncorrected',
                throttle_duration_sec=5.0)
            return 0.0
        d_psi = (yaw_deg(self._observer1_odom.pose.pose.orientation)
                 - yaw_deg(self._observer2_odom.pose.pose.orientation))
        d_psi = (d_psi + 180.0) % 360.0 - 180.0
        self.get_logger().info(
            f'shared_heading: {self.observer1_name} yaw slaved to {self.observer2_name}, '
            f'delta={d_psi:+.2f} deg',
            throttle_duration_sec=5.0)
        return d_psi

    def _bearing_bias_for(self, source_name: str, stamp_ns: int) -> Tuple[float, float]:
        """Total (az, el) bias removed from this observer's rays, degrees.

        The configured per-boot bias, plus for observer1 under shared_heading the live yaw
        difference to observer2. Azimuth offsets about world Z compose additively, so a single
        derotate_bearing call applies both. One accessor, because the debug overlay has to
        rebias by exactly what was removed or the drawn ray lands a bias away from the target.
        """
        az_deg, el_deg = self.bearing_bias.get(source_name, (0.0, 0.0))
        if source_name == self.observer1_name:
            az_deg += self._shared_heading_az_deg(stamp_ns)
        return az_deg, el_deg

    def _apply_bearing_bias(self, obs: SpatialObservation) -> SpatialObservation:
        """Cancel this observer's known constant bearing bias, if one is configured."""
        az_deg, el_deg = self._bearing_bias_for(obs.source_name, obs.stamp_ns)
        if not az_deg and not el_deg:
            return obs
        return dataclasses.replace(obs, ray=derotate_bearing(obs.ray, az_deg, el_deg))

    def _collect_bias_sample(self, obs: SpatialObservation) -> None:
        """Record one ray's error against GT. Needs no pair, so it works on any bag."""
        p_gt = self.gt_position_in_target_frame()
        if p_gt is None:
            return
        az, el = bearing_error_az_el_deg(obs.origin, obs.ray, p_gt)
        # The true bearing to GT is the regressor fit_field_slope needs; storing it here
        # costs one arctan2 per observation and keeps the fit free of any tf lookup.
        # Range comes along because it is what separates an angular fault from a metric one:
        # a wrong camera pitch holds its degrees at every range, a wrong origin height holds
        # its metres and its angle decays as 1/range. See differential_bias().
        self._bias_samples[obs.source_name].append(
            (obs.stamp_ns, az, el, *_bearing_az_el_deg(obs.origin, p_gt),
             float(np.linalg.norm(p_gt - obs.origin))))
        # Refit on a stride: the log is throttled but the fit is not, and it is O(n) over a
        # list that now grows per observation rather than per pair.
        self._bias_sample_count += 1
        if self._bias_sample_count % 50 == 0:
            self._log_bias_fit()

    def _log_bias_fit(self) -> None:
        """Report the constant az/el bias each observer's rays carry against GT.

        Copy `az`/`el` straight into bearing_bias.<role> in the config to cancel a
        miscalibrated compass. Only meaningful while the offset is genuinely constant, which
        the held-out `res` verifies: a res near zero means constant, a large one means drift.
        """
        parts = []
        for name, samples in sorted(self._bias_samples.items()):
            fit = fit_bias_offsets(samples)
            if fit is None:
                continue
            az_fit, el_fit, az_res, el_res, n = fit
            parts.append(
                f'{name} n={n} az={az_fit:+.2f}deg el={el_fit:+.2f}deg '
                f'(holdout res az={az_res:+.2f} el={el_res:+.2f})'
            )
            field = fit_field_slope(samples)
            if field is not None:
                el, az = field['el'], field['az']
                parts[-1] += (
                    f"""\n           el_err: d/d(el)={el['d_el']:+.3f} """
                    f"""d/d(az)={el['d_az']:+.3f}<-ROLL drift={el['drift']:+.2f}/min """
                    f"""range=[{el['lo']:+.1f},{el['hi']:+.1f}]deg\n"""
                    f"""           az_err: d/d(el)={az['d_el']:+.3f} """
                    f"""d/d(az)={az['d_az']:+.3f} drift={az['drift']:+.2f}/min """
                    f"""range=[{az['lo']:+.1f},{az['hi']:+.1f}]deg\n"""
                    f"""           leverage: el span={field['el_span']:.1f}deg """
                    f"""(corr w/ t {field['corr_el_t']:.2f}), az span={field['az_span']:.1f}deg """
                    f"""(corr w/ t {field['corr_az_t']:.2f})"""
                )
            # Same angular-vs-metric question asked of this observer alone. The differential
            # only answers it when the two observers disagree; when one of them carries the
            # whole fault, as uav81 does on the v3 flight, the difference is the wrong probe.
            arr = np.array(samples, dtype=np.float64)
            m = metric_or_angular(arr[:, 2], arr[:, 5])
            parts[-1] += (
                f'\n           shape: el={m[0]:+.2f}deg = {m[1]:+.2f} m at target | '
                f'residual angular={m[2]:.2f} m metric={m[3]:.2f} m '
                f'over range [{m[4]:.1f},{m[5]:.1f}] m -> '
                + verdict_metric_or_angular(m[2], m[3], m[4], m[5], m[1], name)
            )
        # The absolute el error above is contaminated by the GT's own altitude error, which
        # both observers share. Differencing the two cancels it whenever they sit at nearly
        # the same range, so a metric GT error subtracts out and the RELATIVE geometry
        # survives. Measured on the v3 flight: baseline 9.09 m, range 6..49 m, parallax 29 deg
        # median -- a healthy geometry, but the range ratio swings 0.53..2.08, so the
        # same-range precondition holds only in the middle and is enforced in the difference.
        # Samples are now one per observation, so the two lists differ in length and are
        # matched by nearest timestamp within the same window the pairer uses.
        s1 = self._bias_samples.get(self.observer1_name) or []
        s2 = self._bias_samples.get(self.observer2_name) or []
        if parts:
            d = differential_bias(s1, s2, self.max_pair_age_ns)
            if d is not None:
                n_d, d_az, d_el, d_m, res_ang, res_met, r_lo, r_hi = d
                # Name the better-fitting model, but only once the range span is wide enough
                # for the two to be distinguishable at all.
                verdict = verdict_metric_or_angular(res_ang, res_met, r_lo, r_hi, d_m, 'obs2 vs obs1')
                parts.append(
                    f'diff(obs2-obs1) n={n_d} az={d_az:+.2f}deg el={d_el:+.2f}deg '
                    f'({d_m:+.2f} m at target) <- GT-free\n'
                    f'           residual on same axis: angular={res_ang:.2f} m '
                    f'metric={res_met:.2f} m over range [{r_lo:.1f},{r_hi:.1f}] m'
                    f' -> {verdict}')
        if parts:
            # One log call for every observer: throttling is keyed on the call site, so a
            # per-observer call would let the first name starve all the others.
            self.get_logger().info(f'[BIASFIT] ' + ' || '.join(parts), throttle_duration_sec=15.0)

    def gt_position_in_target_frame(self) -> Optional[np.ndarray]:
        """Latest target ground truth expressed in the triangulation target frame.

        The GT odometry is published in the TARGET drone's own origin frame, which carries a
        different Z datum than observer1's. Comparing it raw put a fixed several-metre bias on
        every altitude metric and on the bearing check, so it goes through the same utm hop
        as the remote observations.
        """
        if self.target_gt_odom is None:
            return None
        p = self.target_gt_odom.pose.pose.position
        pos = np.array([p.x, p.y, p.z], dtype=np.float64)
        frame_id = self.target_gt_odom.header.frame_id
        if not frame_id or frame_id == self.target_frame_id:
            return pos
        datum = self._datum_rt(frame_id, stamp_to_ns(self.target_gt_odom.header.stamp), 'target GT')
        if datum is None:
            return None
        rot, trans = datum
        return rot @ pos + trans + np.array([0.0, 0.0, self.target_gt_z_offset_m])

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
                depth_valid=obs.depth_valid,
                depth_m=obs.depth_m,
                depth_sigma_m=obs.depth_sigma_m,
            )

        # Route a remote observer through utm_origin. Its <uav>/world_origin shares orientation with the
        # target frame (all MRS origin frames are north-aligned) but carries a per-flight position/Z datum
        # offset: each drone inits its own takeoff altitude, so the two world_origins can sit several metres
        # apart in z (measured 5.4 m in one flight, 0.19 m in another). A direct target<-source lookup fails
        # because the two drones' tf trees are disconnected (each rooted at its own fcu). So hop through utm,
        # the globally-consistent grid, in two INTRA-tree lookups.
        # ponytail: assumes both drones share one UTM zone (same physical utm grid); a cross-zone flight
        # would need an explicit utm-to-utm bridge transform.
        datum = self._datum_rt(obs.frame_id, obs.stamp_ns, obs.source_name)
        if datum is None:
            return None
        rot, trans = datum
        origin = rot @ obs.origin + trans
        ray = normalize(rot @ obs.ray)

        return SpatialObservation(
            stamp_ns=obs.stamp_ns,
            origin=origin,
            ray=ray,
            frame_id=self.target_frame_id,
            source_name=obs.source_name,
            observation_id=obs.observation_id,
            confidence=obs.confidence,
            tf_delta_ns=obs.tf_delta_ns,
            tf_status=obs.tf_status,
            depth_valid=obs.depth_valid,
            depth_m=obs.depth_m,
            depth_sigma_m=obs.depth_sigma_m,
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

    def _probe_reset(self) -> None:
        self._probe = {
            'best_gap': None, 'best_gap_dt_ms': None, 'best_gap_baseline': None,
            'min_dt_ms': None, 'min_dt_gap': None,
            'feasible_pairs': 0, 'cycles': 0,
        }

    def _probe_reject_geometry(self) -> None:
        """Measure raw ray geometry of a failing cycle's time-feasible pairs.

        For every candidate pair within the time window, compute the RAW ray-gap
        (no time-alignment, so it reflects pure ray geometry) and its dt. Keep the
        window's smallest achievable gap and the smallest-dt pair's gap. Reading:
        best_gap large even at small dt => geometry/calibration; a small gap only
        at small dt => time-sync (rays for different target instants). Runs only on
        failing cycles and the buffers are small, so the O(n*m) scan is cheap.
        """
        best_gap = min_dt_ms = None
        best_gap_dt_ms = best_gap_baseline = min_dt_gap = None
        n_feasible = 0
        for obs1 in self.observer1_observations:
            for obs2 in self.observer2_observations:
                dt_ns = abs(obs1.stamp_ns - obs2.stamp_ns)
                if dt_ns > self.max_pair_age_ns:
                    continue
                n_feasible += 1
                dt_ms = dt_ns / 1e6
                baseline = float(np.linalg.norm(obs2.origin - obs1.origin))
                _, gap = triangulate_midpoint(obs1.origin, obs1.ray, obs2.origin, obs2.ray)
                if best_gap is None or gap < best_gap:
                    best_gap, best_gap_dt_ms, best_gap_baseline = gap, dt_ms, baseline
                if min_dt_ms is None or dt_ms < min_dt_ms:
                    min_dt_ms, min_dt_gap = dt_ms, gap
        if n_feasible == 0:
            return
        p = self._probe
        if p['best_gap'] is None or best_gap < p['best_gap']:
            p['best_gap'], p['best_gap_dt_ms'], p['best_gap_baseline'] = (
                best_gap, best_gap_dt_ms, best_gap_baseline)
        if p['min_dt_ms'] is None or min_dt_ms < p['min_dt_ms']:
            p['min_dt_ms'], p['min_dt_gap'] = min_dt_ms, min_dt_gap
        p['feasible_pairs'] += n_feasible
        p['cycles'] += 1

    def _log_pairing_diagnostics(self, now_ns: int) -> None:
        """Emit a one-line pairing/gating summary every 2 s, then reset counters.

        Answers at a glance: are rays arriving (rx), is the second observer
        starved (nopair_starved_*), do rays arrive but never overlap/cross
        (nopair_no_overlap_or_geom), or are formed pairs rejected downstream
        (drop_baseline / drop_raygap / drop_wls / drop_maha) vs accepted.
        """
        if self._diag_last_log_ns == 0:
            self._diag_last_log_ns = now_ns
            return
        if (now_ns - self._diag_last_log_ns) < 2_000_000_000:
            return
        self._diag_last_log_ns = now_ns
        d = self._diag
        self.get_logger().info(
            'PAIRING 2s | rx: {o1}={a} {o2}={b} | nopair: starved_{o1}={c} '
            'starved_{o2}={e} no_overlap_or_geom={f} | drops: baseline={g} '
            'raygap={h} wls={i} maha={j} | accepted={k}'.format(
                o1=self.observer1_name, o2=self.observer2_name,
                a=d['rx_' + self.observer1_name], b=d['rx_' + self.observer2_name],
                c=d['nopair_starved_obs1'], e=d['nopair_starved_obs2'],
                f=d['nopair_no_overlap_or_geom'], g=d['drop_baseline'],
                h=d['drop_raygap'], i=d['drop_wls_error'], j=d['drop_mahalanobis'],
                k=d['accepted'],
            )
        )
        self.get_logger().info(
            'DBGIMG 2s | pub: stereo={s} mono={m} | skip: no_obs={a} no_frame={b} dedup={c}'.format(
                s=d['dbg_pub_stereo'], m=d['dbg_pub_mono'],
                a=d['dbg_no_obs'], b=d['dbg_no_frame'], c=d['dbg_dedup'],
            )
        )
        p = self._probe
        if p['cycles']:
            self.get_logger().info(
                'GEOM-PROBE 2s | feasible_pairs={fp} over {cy} failing cycles | '
                'best_gap={bg:.2f} m @ dt={bd:.1f} ms (baseline={bb:.2f} m) | '
                'smallest_dt={md:.1f} ms @ gap={mg:.2f} m | gate={gate:.2f} m'.format(
                    fp=p['feasible_pairs'], cy=p['cycles'],
                    bg=p['best_gap'], bd=p['best_gap_dt_ms'], bb=p['best_gap_baseline'],
                    md=p['min_dt_ms'], mg=p['min_dt_gap'],
                    gate=self.max_triangulation_error_m,
                )
            )
        self._diag.clear()
        self._probe_reset()

    def processing_timer_callback(self) -> None:
        """Timer callback: handles single-ray fallback and coasting only.

        Stereo triangulation is intentionally event-driven (triggered on every
        incoming observation).  The timer exists solely for two purposes:
          1. Single RGB-D fallback  - when stereo pairing cannot be formed but
             at least one observer has a fresh, valid depth measurement.
          2. Coasting               - pure kinematic prediction when no sensor
             data is available at all.

        This separation eliminates the race condition where the timer fires
        between the arrival of obs1 and obs2 from the two cameras, which would
        cause try_match_and_triangulate() to see an empty buffer and
        prematurely fall through to Single-Ray or Coast.
        """
        now_ns = self.get_clock().now().nanoseconds
        self._prune_stale_observations(now_ns)
        self._log_pairing_diagnostics(now_ns)

        # Guard: if the filter has consumed a measurement very recently it is
        # healthy — skip fallback evaluation to avoid double-updating.
        #
        # Both terms MUST be on the same clock.  imm_last_stamp_ns is the image
        # CAPTURE time; differencing it against the wallclock 'now' folds the
        # whole pipeline latency (JPEG -> WiFi -> YOLO on CPU -> WiFi) into the
        # age, which alone exceeds any sane threshold and makes this guard fire
        # on every tick.  _last_measurement_wall_ns records when the update was
        # consumed, so the subtraction below is wallclock - wallclock.
        if self._last_measurement_wall_ns:
            dt_since_last = (now_ns - self._last_measurement_wall_ns) / 1e9
        else:
            dt_since_last = float('inf')

        # Only evaluate fallback if no measurement has been consumed recently.
        # The threshold is tied to the DETECTION rate, not the camera rate: the
        # filter is starved only once a detection period has gone by unfed.
        if dt_since_last < self.coast_after_sec:
            return

        # Priority 2: SINGLE RGB-D FALLBACK
        best_obs = None
        for buffer in (self.observer1_observations, self.observer2_observations):
            for obs in reversed(buffer):
                if obs.depth_valid and obs.confidence >= self.single_obs_min_confidence:
                    age_ns = abs(now_ns - obs.stamp_ns)
                    if age_ns <= self.single_obs_max_age_ns:
                        if best_obs is None or obs.stamp_ns > best_obs.stamp_ns:
                            best_obs = obs

        if best_obs is not None:
            if self.execute_single_observation_fusion(best_obs, best_obs.stamp_ns):
                self._publish_fusion_mode()
                return

        # Priority 3: COAST — only when no sensor data is available at all.
        if self.imm_initialized and self.imm_last_stamp_ns is not None:
            if dt_since_last > self.coast_after_sec:
                self._coast_predict(now_ns)
                self.last_fusion_mode = "COAST"
                self._publish_fusion_mode()
                
    def _publish_fusion_mode(self) -> None:
        msg = String()
        msg.data = self.last_fusion_mode
        self.fusion_mode_pub.publish(msg)

    def _predict_position_at_stamp_ns(self, stamp_ns: int) -> np.ndarray:
        """Extrapolate the filtered target position to an arbitrary timestamp."""
        if not self.imm_initialized or self.imm_last_stamp_ns is None:
            return self.filtered_position.copy()

        dt = (stamp_ns - self.imm_last_stamp_ns) / 1e9
        return (
            self.filtered_position
            + self.filtered_velocity * dt
            + 0.5 * self.filtered_acceleration * dt * dt
        )

    def _select_best_observation_pair(
        self,
    ) -> Optional[Tuple[SpatialObservation, SpatialObservation, int]]:
        """Pick one feasible observer pair using legacy or prediction-guided association."""
        if not self.pairing_prediction_enabled:
            return select_timestamp_only_pair(
                list(self.observer1_observations),
                list(self.observer2_observations),
                self.max_pair_age_ns,
            )

        best_obs1 = None
        best_obs2 = None
        best_diff = self.max_pair_age_ns + 1
        best_cost = float('inf')
        best_geom = float('inf')
        best_score: Optional[PredictionGuidedPairScore] = None

        for obs1 in self.observer1_observations:
            for obs2 in self.observer2_observations:
                pair_diff_ns = abs(obs1.stamp_ns - obs2.stamp_ns)
                if pair_diff_ns > self.max_pair_age_ns:
                    continue

                aligned_obs1, aligned_obs2, aligned_stamp_ns = self._time_align_observations(obs1, obs2)
                x_pred = self._predict_position_at_stamp_ns(aligned_stamp_ns)
                P_pred_pos = self.combined_cov[:3, :3] if self.imm_initialized else None
                score = score_prediction_guided_pair(
                    aligned_obs1.origin,
                    aligned_obs1.ray,
                    aligned_obs2.origin,
                    aligned_obs2.ray,
                    pair_diff_ns,
                    self.max_pair_age_ns,
                    self.max_triangulation_error_m,
                    self.triangulation_min_baseline_m,
                    self.kf_measurement_noise,
                    x_pred,
                    P_pred_pos,
                    self.pairing_weight_time,
                    self.pairing_weight_geom,
                    self.pairing_weight_innov,
                    self.pairing_mahalanobis_chi2,
                    self.imm_initialized,
                )
                if score is None:
                    continue

                if (
                    score.cost < best_cost
                    or (
                        abs(score.cost - best_cost) <= 1e-12
                        and pair_diff_ns < best_diff
                    )
                    or (
                        abs(score.cost - best_cost) <= 1e-12
                        and pair_diff_ns == best_diff
                        and score.ray_gap_m < best_geom
                    )
                ):
                    best_cost = score.cost
                    best_geom = score.ray_gap_m
                    best_diff = pair_diff_ns
                    best_obs1 = obs1
                    best_obs2 = obs2
                    best_score = score

        if best_obs1 is None or best_obs2 is None:
            return None

        if self.pairing_log_best_cost and best_score is not None:
            self.get_logger().info(
                'Prediction-guided pair selected: '
                f'J={best_score.cost:.3f} '
                f'(c_t={best_score.time_term:.3f}, c_g={best_score.geom_term:.3f}, '
                f'c_i={best_score.innov_term:.3f}, gap={best_score.ray_gap_m:.3f} m, '
                f'dt={best_diff / 1e6:.1f} ms)',
                throttle_duration_sec=1.0,
            )

        return best_obs1, best_obs2, best_diff

    def try_match_and_triangulate(self) -> bool:
        """Pair observer rays from both buffers and triangulate accepted matches."""
        executed = False
        all_pairs = []
        
        while self.observer1_observations and self.observer2_observations:
            selection = self._select_best_observation_pair()
            if selection is None:
                break

            best_obs1, best_obs2, best_diff = selection
            self.observer1_observations.remove(best_obs1)
            self.observer2_observations.remove(best_obs2)
            all_pairs.append((best_obs1, best_obs2, best_diff))

        if not all_pairs:
            # No feasible pair this cycle — record why so the periodic summary can
            # separate "second observer starved" from "rays present but never overlap".
            if not self.observer1_observations:
                self._diag['nopair_starved_obs1'] += 1
            elif not self.observer2_observations:
                self._diag['nopair_starved_obs2'] += 1
            else:
                self._diag['nopair_no_overlap_or_geom'] += 1
                self._probe_reject_geometry()

        # Sort pairs by aligned timestamp to ensure strictly forward-in-time updates
        all_pairs.sort(key=lambda x: max(x[0].stamp_ns, x[1].stamp_ns))

        for best_obs1, best_obs2, best_diff in all_pairs:
            aligned_obs1, aligned_obs2, aligned_stamp_ns = self._time_align_observations(best_obs1, best_obs2)
            accepted = self.execute_triangulation(
                aligned_obs1,
                aligned_obs2,
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
            if accepted:
                self.last_fusion_mode = "STEREO"
                executed = True
                
        return executed

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
                depth_valid=obs.depth_valid,
                depth_m=obs.depth_m,
                depth_sigma_m=obs.depth_sigma_m,
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
            depth_valid=obs.depth_valid,
            depth_m=obs.depth_m,
            depth_sigma_m=obs.depth_sigma_m,
        )

    def execute_single_observation_fusion(
        self,
        obs: SpatialObservation,
        stamp_ns: int,
    ) -> bool:
        """Fuse a single RGB-D observation into the IMM using anisotropic covariance."""
        measurement_3d = obs.origin + obs.ray * obs.depth_m
        R_single = compute_single_ray_measurement_covariance(
            obs.ray,
            obs.depth_m,
            obs.depth_sigma_m,
            self.kf_measurement_noise,
        )

        if not self.imm_initialized:
            self.imm.initialize(measurement_3d, self.kf_initial_covariance * 4.0)
            self.imm_initialized = True
            self.imm_last_stamp_ns = stamp_ns
            self.combined_cov = np.eye(9) * self.kf_initial_covariance * 4.0
            self.filtered_position = measurement_3d.copy()
            self.filtered_velocity = np.zeros(3)
            self.filtered_acceleration = np.zeros(3)
        else:
            dt = (stamp_ns - self.imm_last_stamp_ns) / 1_000_000_000.0
            if dt > 0.0:
                Q_cv = build_process_noise_cv(dt, self.kf_process_noise_vel)
                Q_ca = build_process_noise_ca(dt, self.kf_process_noise_acc)
                x_comb, self.combined_cov, self.imm_probs, accepted, _ = self.imm.predict_and_correct(
                    dt,
                    Q_cv,
                    Q_ca,
                    measurement_3d,
                    R_single,
                    mahalanobis_thresh=self.kf_mahalanobis_threshold,
                )

                if not accepted:
                    self.get_logger().warn(
                        'Single-ray measurement rejected by Mahalanobis gating! Filter coasting.'
                    )
                    return False

                self.filtered_position = x_comb[:3].copy()
                self.filtered_velocity = x_comb[3:6].copy()
                self.filtered_acceleration = x_comb[6:9].copy()

        self.imm_last_stamp_ns = stamp_ns
        self._last_measurement_wall_ns = self.get_clock().now().nanoseconds
        self.last_fusion_mode = "SINGLE_DEPTH"

        raw_msg = PointStamped()
        raw_msg.header.frame_id = self.target_frame_id
        raw_msg.header.stamp.sec = stamp_ns // 1_000_000_000
        raw_msg.header.stamp.nanosec = stamp_ns % 1_000_000_000
        raw_msg.point.x = float(measurement_3d[0])
        raw_msg.point.y = float(measurement_3d[1])
        raw_msg.point.z = float(measurement_3d[2])
        self.triangulation_raw_pub.publish(raw_msg)
        
        return True

    def execute_triangulation(
        self,
        obs1: SpatialObservation,
        obs2: SpatialObservation,
        stamp_ns: int,
    ) -> bool:
        """Triangulate one matched pair, reject weak geometry, and feed the accepted point into the IMM."""
        origin1 = obs1.origin
        dir1 = obs1.ray
        origin2 = obs2.origin
        dir2 = obs2.ray

        baseline = float(np.linalg.norm(origin2 - origin1))
        if baseline < self.triangulation_min_baseline_m:
            self._diag['drop_baseline'] += 1
            return False

        # ── Geometry pre-check (still uses midpoint for the ray-gap test) ────
        # We keep the cheap midpoint solely to gate on max_triangulation_error_m.
        # The WLS call below provides the true MLE estimate and covariance.
        midpoint_check, ray_gap_m = triangulate_midpoint(origin1, dir1, origin2, dir2)

        # DIAGNOSTIC (throttled 5 s): separates a frame/rotation fault from a noise fault.
        #   s,t   = depth of the closest point along each ray. NEGATIVE => the rays meet
        #           BEHIND that observer, i.e. the ray direction is mirrored/rotated.
        #   e1,e2 = angle between each ray and the true bearing observer->GT target.
        #           Both ~0 => geometry is sound. One large => that observer's extrinsics.
        #           Both large and similar => a shared world-frame/heading error.
        s_depth = float(np.dot(midpoint_check - origin1, dir1) / max(np.dot(dir1, dir1), 1e-9))
        t_depth = float(np.dot(midpoint_check - origin2, dir2) / max(np.dot(dir2, dir2), 1e-9))
        gt_txt = ''
        p_gt = self.gt_position_in_target_frame()
        if p_gt is not None:
            # Split each bearing error into azimuth and elevation: a heading bias shows up
            # purely in azimuth, a Z-datum or mount-pitch fault purely in elevation.
            az1, el1 = bearing_error_az_el_deg(origin1, dir1, p_gt)
            az2, el2 = bearing_error_az_el_deg(origin2, dir2, p_gt)
            # NB: bias samples are NOT collected here. Sampling only accepted pairs made the
            # fit unable to bootstrap: a bag whose heading bias is large enough to stop the
            # rays meeting yields no pairs, hence no samples, hence no bias to correct it
            # with. Collection lives in _collect_bias_sample(), one per observation.
            gt_txt = (
                f' | e1={bearing_error_deg(origin1, dir1, p_gt):.1f}deg(az{az1:+.1f},el{el1:+.1f}) '
                f'e2={bearing_error_deg(origin2, dir2, p_gt):.1f}deg(az{az2:+.1f},el{el2:+.1f}) '
                f'gt_d1={np.linalg.norm(p_gt - origin1):.1f}m '
                f'err_mid={np.linalg.norm(midpoint_check - p_gt):.1f}m'
            )
        self.get_logger().info(
            f'[RAYDBG] {obs1.source_name} o=({origin1[0]:.1f},{origin1[1]:.1f},{origin1[2]:.1f}) '
            f'r=({dir1[0]:.3f},{dir1[1]:.3f},{dir1[2]:.3f}) | '
            f'{obs2.source_name} o=({origin2[0]:.1f},{origin2[1]:.1f},{origin2[2]:.1f}) '
            f'r=({dir2[0]:.3f},{dir2[1]:.3f},{dir2[2]:.3f}) | '
            f's={s_depth:.1f}m t={t_depth:.1f}m base={baseline:.1f}m gap={ray_gap_m:.2f}m{gt_txt}',
            throttle_duration_sec=5.0,
        )

        if ray_gap_m > self.max_triangulation_error_m:
            self.get_logger().warn(
                f'Triangulation rejected: ray mismatch {ray_gap_m:.2f} m exceeds {self.max_triangulation_error_m:.2f} m'
            )
            self._diag['drop_raygap'] += 1
            return False

        # ── Observer odometry covariance (for WLS EIV model) ─────────────────
        # When use_odom_uncertainty=True, P_odom_i is fetched and clamped
        # inside triangulate_wls (odom_max_variance). This prevents the
        # barometric Z-blowup (sim: sigma_z ~ 35 m) from inflating R.
        # Reference: Dogancay (2015) Sec. II; Xu et al. (2017) Sec. III.
        P_odom1: Optional[np.ndarray] = None
        P_odom2: Optional[np.ndarray] = None
        if self.use_odom_uncertainty:
            P_odom1 = self._get_observer_position_covariance(
                self._observer1_odom,
                self._observer1_odom_stamp_ns,
                stamp_ns,
                self.observer1_name,
            )
            P_odom2 = self._get_observer_position_covariance(
                self._observer2_odom,
                self._observer2_odom_stamp_ns,
                stamp_ns,
                self.observer2_name,
            )

        # ── Optimal WLS triangulation (MLE + CRLB covariance) ────────────────
        # triangulate_wls() jointly solves for p_hat and R_out in one pass.
        # R_out = Lambda^{-1} is the Cramér-Rao Lower Bound for this geometry
        # and is passed directly as R_k to the IMM-KF measurement update.
        #
        # IV-WLS: Pass the filtered position to break the Pseudo-Linear Estimator bias
        prior_pos = self.filtered_position if self.imm_initialized else None

        try:
            measurement, R = triangulate_wls(
                origin1, dir1, P_odom1,
                origin2, dir2, P_odom2,
                var_px=self.kf_measurement_noise,
                use_odom_uncertainty=self.use_odom_uncertainty,
                distance_dependent_noise=self.distance_dependent_noise,
                odom_max_variance=self.odom_max_variance,
                anisotropic_bearing_noise=self.anisotropic_bearing_noise,
                bearing_var_az=self.bearing_var_az,
                bearing_var_el=self.bearing_var_el,
                prior_target_pos=prior_pos,
                normalize_crlb=self.normalize_crlb,
                logger=self.get_logger(),
            )
        except ValueError as exc:
            self.get_logger().warn(str(exc), throttle_duration_sec=2.0)
            self._diag['drop_wls_error'] += 1
            return False

        # Publish raw WLS estimate (MLE, before IMM smoothing)
        raw_msg = PointStamped()
        raw_msg.header.frame_id = self.target_frame_id
        raw_msg.header.stamp.sec = stamp_ns // 1_000_000_000
        raw_msg.header.stamp.nanosec = stamp_ns % 1_000_000_000
        raw_msg.point.x = float(measurement[0])
        raw_msg.point.y = float(measurement[1])
        raw_msg.point.z = float(measurement[2])
        self.triangulation_raw_pub.publish(raw_msg)

        if not self.imm_initialized:
            self.imm.initialize(measurement, self.kf_initial_covariance)
            self.imm_initialized = True
            self.imm_last_stamp_ns = stamp_ns
            self._last_measurement_wall_ns = self.get_clock().now().nanoseconds
            self.combined_cov = np.eye(9) * self.kf_initial_covariance
            self.filtered_position = measurement.copy()
            self.filtered_velocity = np.zeros(3)
            self.filtered_acceleration = np.zeros(3)
        else:
            dt = (stamp_ns - self.imm_last_stamp_ns) / 1_000_000_000.0
            if dt > 0.0:
                Q_cv = build_process_noise_cv(dt, self.kf_process_noise_vel)
                Q_ca = build_process_noise_ca(dt, self.kf_process_noise_acc)

                x_comb, self.combined_cov, self.imm_probs, accepted, _ = self.imm.predict_and_correct(
                    dt,
                    Q_cv,
                    Q_ca,
                    measurement,
                    R,
                    mahalanobis_thresh=self.kf_mahalanobis_threshold,
                )

                if not accepted:
                    self._diag['drop_mahalanobis'] += 1
                    self.get_logger().warn(
                        'Outlier measurement rejected by Mahalanobis gating! Filter coasting purely on kinetics.'
                    )
                
                # --- SEQUENTIAL 1D DEPTH FUSION ---
                if accepted and self.df_use_range_constraint:
                    for obs in (obs1, obs2):
                        if obs.depth_valid and obs.depth_m > 0:
                            # The target vector from camera to the currently estimated target position
                            # in world coordinates.
                            diff = x_comb[:3] - obs.origin
                            dist = float(np.linalg.norm(diff))
                            if dist > 0.1:
                                h_pos = diff / dist
                                
                                h_cv = np.zeros(6, dtype=np.float64)
                                h_cv[:3] = h_pos
                                
                                h_ca = np.zeros(9, dtype=np.float64)
                                h_ca[:3] = h_pos
                                
                                z_1d = obs.depth_m
                                r_1d = obs.depth_sigma_m ** 2
                                
                                x_comb, self.combined_cov, self.imm_probs, depth_accepted = self.imm.correct_1d(
                                    z_scalar=z_1d,
                                    h_vec_cv=h_cv,
                                    h_vec_ca=h_ca,
                                    r_scalar=r_1d,
                                    chi2_thresh=self.df_range_chi2,
                                )
                                
                                if not depth_accepted:
                                    self.get_logger().warn(
                                        f'Depth constraint rejected by Mahalanobis gating! '
                                        f'(z={z_1d:.2f}m, expected={dist:.2f}m)'
                                    )

                self.filtered_position = x_comb[:3].copy()
                self.filtered_velocity = x_comb[3:6].copy()
                self.filtered_acceleration = x_comb[6:9].copy()
                
                # Clamp velocity to max limits (e.g., F450 limit 10m/s)
                speed = np.linalg.norm(self.filtered_velocity)
                if speed > 10.0:
                    self.filtered_velocity = (self.filtered_velocity / speed) * 10.0
                    self.imm.sc_cv.x[3:6] = self.filtered_velocity
                    self.imm.sc_ca.x[3:6] = self.filtered_velocity
                    x_comb[3:6] = self.filtered_velocity
                    
                self.imm_last_stamp_ns = stamp_ns
                self._last_measurement_wall_ns = self.get_clock().now().nanoseconds

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
        self._diag['accepted'] += 1

        # Publish evaluation packet for CMA-ES offline tuning
        if self.triangulation_evaluation_pub is not None:
            payload = {
                'schema_version': 1,
                'frame_stamp_ns': stamp_ns,
                'triangulation_residual_m': float(ray_gap_m),
                'num_observers': 2,
                'estimated_position_xyz': self.filtered_position.tolist(),
                'estimated_velocity_xyz': self.filtered_velocity.tolist(),
                'covariance_6x6': self.combined_cov[:6, :6].tolist(),
            }
            packet_msg = String()
            packet_msg.data = json.dumps(payload, separators=(',', ':'), ensure_ascii=True)
            self.triangulation_evaluation_pub.publish(packet_msg)

        return True

    def _coast_predict(self, now_ns: int) -> None:
        """Advance the filter without a measurement when the matched observation stream goes temporarily quiet.

        CRITICAL: this method intentionally does NOT update self.imm_last_stamp_ns.
        imm_last_stamp_ns must always reflect the timestamp of the last real
        sensor measurement (stereo triangulation or single-ray fusion).  Updating
        it here with the wallclock would cause the next genuine measurement — whose
        stamp_ns is in sensor time, typically behind wallclock — to compute a
        negative dt and be silently discarded by the `if dt > 0.0` guard inside
        execute_triangulation / execute_single_observation_fusion.
        """
        if self.imm_last_stamp_ns is None:
            return
        dt = (now_ns - self.imm_last_stamp_ns) / 1e9
        if dt <= 0.0 or dt > 5.0:
            return

        tau = 0.5  # Singer decay time constant
        alpha = np.exp(-dt / tau)

        # Project position/velocity/acceleration forward by dt, but store the
        # result only in the publishing fields — do NOT advance imm_last_stamp_ns.
        coast_position = (
            self.filtered_position
            + self.filtered_velocity * dt
            + 0.5 * self.filtered_acceleration * dt * dt
        )
        coast_velocity = self.filtered_velocity + self.filtered_acceleration * dt
        coast_acceleration = self.filtered_acceleration * alpha

        # Clamp maximum speed (e.g., F450 limit ~10m/s)
        speed = np.linalg.norm(coast_velocity)
        if speed > 10.0:
            coast_velocity = (coast_velocity / speed) * 10.0

        self.get_logger().info(f'Coast predict: dt={dt:.2f}s, no measurement', throttle_duration_sec=2.0)
        self.publish_pose_estimate(
            coast_position,
            coast_velocity,
            coast_acceleration,
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

            # Clamp acceleration magnitude to a physically achievable envelope (~2g for racing drones).
            MAX_ACCEL_PRED = 20.0  # m/s²
            acc_mag = float(np.linalg.norm(acc))
            if acc_mag > MAX_ACCEL_PRED:
                acc = acc * (MAX_ACCEL_PRED / acc_mag)

            # Singer model decay: acceleration decays to zero over time, naturally
            # modeling the Markov transition back to Constant Velocity (straight flight).
            # tau=1.5s allows the prediction to follow the full arc of a banking turn
            # before decaying to straight flight (tau=0.5s was too short and produced
            # trajectories that appeared stuck on CV during sustained maneuvers).
            tau = 1.5
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

        gt_position = self.gt_position_in_target_frame()
        if gt_position is not None:
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
                gt.pose.position.x = float(gt_position[0])
                gt.pose.position.y = float(gt_position[1])
                gt.pose.position.z = float(gt_position[2])
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

        gt_position = self.gt_position_in_target_frame()
        if gt_position is None:
            return
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