#!/usr/bin/env python3
"""
Live Monitor — runs on your PC, subscribes to drone topics over WiFi.

Usage:
    python3 live_monitor.py

It reads runtime_stack.yaml to auto-discover UAV names and the odom suffix,
as well as the display mode.
Press CTRL+C or close the window to quit.
"""

import threading
import time as _time
from pathlib import Path as FsPath
from collections import deque
from typing import Optional

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path


# ── Configuration ────────────────────────────────────────────────────────────

MAX_TRAIL = 15000         # max points kept in rolling buffers
UPDATE_INTERVAL_MS = 150  # plot refresh rate (~7 Hz)

# ── Theme ────────────────────────────────────────────────────────────────────
_BG      = '#0f111a'
_PANEL   = '#1a1c2e'
_GRID    = '#2a2d45'
_TEXT    = '#c8cad0'
_ACCENT  = '#5c6bc0'
_GT      = '#4fc3f7'
_EST     = '#ff8a65'
_PRED    = '#ef5350'
_OBS1    = '#ce93d8'
_OBS2    = '#ffd54f'
_ERR     = '#e57373'


class LiveMonitorNode(Node):
    def __init__(self) -> None:
        super().__init__('live_monitor')

        # ── Load config from runtime_stack.yaml ──────────────────────────
        script_dir = FsPath(__file__).resolve().parent
        config_path = script_dir.parent / 'config' / 'runtime_stack.yaml'

        obs1 = 'uav1'
        obs2 = 'uav2'
        target = 'uav3'
        odom_suffix = 'hw_api/ground_truth'
        gt_aligned_topic = ''      # empty -> fall back to the target's raw odom topic
        self.mode = 'classic'

        if config_path.exists():
            import yaml
            try:
                cfg = yaml.safe_load(open(config_path, 'r'))
                if cfg:
                    roles = cfg.get('uav_roles', {})
                    obs1 = roles.get('observer1', obs1)
                    obs2 = roles.get('observer2', obs2)
                    target = roles.get('target', target)
                    eval_cfg = cfg.get('evaluation', {})
                    odom_suffix = eval_cfg.get('odom_topic_suffix', odom_suffix)
                    gt_aligned_topic = eval_cfg.get('gt_aligned_topic', gt_aligned_topic)
                    self.mode = eval_cfg.get('live_monitor_mode', self.mode)
            except Exception as e:
                self.get_logger().warn(f'Could not load config: {e}')

        # ── Topics ───────────────────────────────────────────────────────
        # Aligned GT when the triangulation node republishes it: the raw topic is in the target
        # drone's own origin frame and its z datum differs from the estimate's by several metres,
        # which the Z panel would draw as a constant tracking error.
        self.target_odom_topic = gt_aligned_topic or f'/{target}/{odom_suffix}'
        self.estimate_odom_topic = '/target/odometry_estimate'
        self.predicted_path_topic = '/target/predicted_trajectory'

        self.obs1_odom_topic = f'/{obs1}/{odom_suffix}'
        self.obs2_odom_topic = f'/{obs2}/{odom_suffix}'

        self.get_logger().info(f'Target odom:       {self.target_odom_topic}')
        self.get_logger().info(f'Estimate odom:     {self.estimate_odom_topic}')
        self.get_logger().info(f'Predicted path:    {self.predicted_path_topic}')
        self.get_logger().info(f'Observer 1 odom:   {self.obs1_odom_topic}')
        self.get_logger().info(f'Observer 2 odom:   {self.obs2_odom_topic}')
        self.get_logger().info(f'Live Monitor Mode: {self.mode}')

        # ── Data buffers (thread-safe via GIL for simple appends) ────────
        self.gt_data = deque(maxlen=MAX_TRAIL)
        self.est_data = deque(maxlen=MAX_TRAIL)
        self.err_data = deque(maxlen=MAX_TRAIL)
        self.err_xy_data = deque(maxlen=MAX_TRAIL)
        self.err_z_data = deque(maxlen=MAX_TRAIL)
        self.pred_xy: Optional[np.ndarray] = None   # latest predicted path
        self.obs1_pos: Optional[np.ndarray] = None
        self.obs2_pos: Optional[np.ndarray] = None
        self._gt_count = 0
        self._est_count = 0
        self._t0 = _time.monotonic()
        self._first_ros_time = None
        self.lock = threading.Lock()

        # ── Subscribers ──────────────────────────────────────────────────
        self.create_subscription(Odometry, self.target_odom_topic, self._gt_cb, 20)
        self.create_subscription(Odometry, self.estimate_odom_topic, self._est_cb, 20)
        self.create_subscription(Path, self.predicted_path_topic, self._pred_cb, 10)
        self.create_subscription(Odometry, self.obs1_odom_topic, self._obs1_cb, 10)
        self.create_subscription(Odometry, self.obs2_odom_topic, self._obs2_cb, 10)

    # ── Callbacks ────────────────────────────────────────────────────────
    def _time_from_msg(self, msg) -> float:
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self._first_ros_time is None:
            self._first_ros_time = t
        return t - self._first_ros_time

    def _pos(self, msg: Odometry) -> np.ndarray:
        p = msg.pose.pose.position
        return np.array([p.x, p.y, p.z])

    def _nearest_estimate_position(self, target_time: float) -> Optional[np.ndarray]:
        if not self.est_data:
            return None

        best_position = None
        best_dt = float('inf')
        for estimate_time, estimate_position in self.est_data:
            dt = abs(estimate_time - target_time)
            if dt < best_dt:
                best_dt = dt
                best_position = estimate_position

        return None if best_position is None else best_position.copy()

    def _gt_cb(self, msg: Odometry):
        pos = self._pos(msg)
        t = self._time_from_msg(msg)
        with self.lock:
            self.gt_data.append((t, pos))
            self._gt_count += 1
            nearest_estimate = self._nearest_estimate_position(t)
            if nearest_estimate is not None:
                err_3d = float(np.linalg.norm(pos - nearest_estimate))
                err_xy = float(np.linalg.norm(pos[:2] - nearest_estimate[:2]))
                err_z = float(abs(pos[2] - nearest_estimate[2]))
                self.err_data.append((t, err_3d))
                self.err_xy_data.append((t, err_xy))
                self.err_z_data.append((t, err_z))

    def _est_cb(self, msg: Odometry):
        t = self._time_from_msg(msg)
        with self.lock:
            self.est_data.append((t, self._pos(msg)))
            self._est_count += 1

    def _pred_cb(self, msg: Path):
        if msg.poses:
            with self.lock:
                self.pred_xy = np.array([
                    [p.pose.position.x, p.pose.position.y, p.pose.position.z]
                    for p in msg.poses
                ])

    def _obs1_cb(self, msg: Odometry):
        with self.lock:
            self.obs1_pos = self._pos(msg)

    def _obs2_cb(self, msg: Odometry):
        with self.lock:
            self.obs2_pos = self._pos(msg)


def zoom_factory(ax, base_scale=1.2):
    ax._zoom_factor = 1.0
    def zoom(event):
        if event.inaxes != ax: return
        if event.button == 'up':
            ax._zoom_factor /= base_scale
        elif event.button == 'down':
            ax._zoom_factor *= base_scale
    
    fig = ax.get_figure()
    fig.canvas.mpl_connect('scroll_event', zoom)
    return zoom


def main():
    rclpy.init()
    node = LiveMonitorNode()

    # ── Spin ROS in background thread ────────────────────────────────────
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    # ── Matplotlib setup ─────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 8))
    fig.patch.set_facecolor(_BG)
    
    fig.suptitle(f'VISION STACK \u2014 Live Monitor ({node.mode.upper()} MODE)', fontsize=15, fontweight='bold',
                 color='white', fontfamily='monospace')

    def _style_ax(ax, title, xlabel, ylabel):
        ax.set_facecolor(_PANEL)
        ax.set_title(title, color='white', fontsize=10, fontweight='bold', pad=8)
        ax.set_xlabel(xlabel, color=_TEXT, fontsize=8)
        ax.set_ylabel(ylabel, color=_TEXT, fontsize=8)
        ax.tick_params(colors=_TEXT, labelsize=7)
        ax.grid(True, alpha=0.35, color=_GRID, linewidth=0.5)
        ax.minorticks_on()
        ax.grid(True, which='minor', alpha=0.12, color=_GRID, linewidth=0.3)
        for spine in ax.spines.values():
            spine.set_color(_GRID)

    def _style_ax3d(ax, title):
        ax.set_facecolor(_BG)
        ax.set_title(title, color='white', fontsize=11, fontweight='bold', pad=10)
        ax.set_xlabel('East [m]', color=_TEXT, fontsize=8, labelpad=8)
        ax.set_ylabel('North [m]', color=_TEXT, fontsize=8, labelpad=8)
        ax.set_zlabel('Altitude Z [m]', color=_TEXT, fontsize=8, labelpad=8)
        ax.tick_params(colors=_TEXT, labelsize=7)
        
        # Transparent panes for a more modern, cleaner look
        ax.xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
        ax.yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
        ax.zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
        
        # Subtle grid lines
        ax.xaxis._axinfo["grid"].update({"color": _GRID, "linewidth": 0.5, "alpha": 0.5})
        ax.yaxis._axinfo["grid"].update({"color": _GRID, "linewidth": 0.5, "alpha": 0.5})
        ax.zaxis._axinfo["grid"].update({"color": _GRID, "linewidth": 0.5, "alpha": 0.5})
        
        # Hide axis spines
        ax.xaxis.line.set_color((0.0, 0.0, 0.0, 0.0))
        ax.yaxis.line.set_color((0.0, 0.0, 0.0, 0.0))
        ax.zaxis.line.set_color((0.0, 0.0, 0.0, 0.0))

        # Enforce equal aspect ratio so rotations don't skew
        if hasattr(ax, 'set_box_aspect'):
            ax.set_box_aspect((1, 1, 1))

    # Initialize possible axes to None
    ax_xy, ax_z, ax_err, ax_err_xy, ax_err_z, ax_3d = None, None, None, None, None, None
    ax_bar = None
    
    if node.mode == 'full_3d':
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.08], hspace=0.35, wspace=0.25,
                              left=0.05, right=0.97, top=0.92, bottom=0.06)
        ax_xy  = fig.add_subplot(gs[0, 0])
        ax_3d  = fig.add_subplot(gs[0, 1], projection='3d')
        ax_z   = fig.add_subplot(gs[1, 0])
        ax_err = fig.add_subplot(gs[1, 1])
        ax_bar = fig.add_subplot(gs[2, :])
    elif node.mode == 'minimal_3d':
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 0.06], hspace=0.15,
                              left=0.05, right=0.97, top=0.92, bottom=0.06)
        ax_3d = fig.add_subplot(gs[0, 0], projection='3d')
        ax_bar = fig.add_subplot(gs[1, 0])
    else: # classic
        gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.08], hspace=0.35, wspace=0.3,
                              left=0.05, right=0.97, top=0.92, bottom=0.06)
        ax_xy  = fig.add_subplot(gs[0, 0])
        ax_z   = fig.add_subplot(gs[0, 1])
        ax_err = fig.add_subplot(gs[0, 2])
        ax_bar = fig.add_subplot(gs[1, :])

    # Objects to animate
    ln_gt, ln_est, ln_pred = None, None, None
    mk_obs1, mk_obs2, mk_tgt, mk_est = None, None, None, None
    
    ln_gt_z, ln_est_z = None, None
    ln_err, ln_avg = None, None
    ln_err_xy, ln_avg_xy = None, None
    ln_err_z, ln_avg_z = None, None
    
    ln3_gt, ln3_est, ln3_pred = None, None, None
    mk3_obs1, mk3_obs2, mk3_tgt, mk3_est = None, None, None, None
    
    # ── Setup XY bird's eye ──
    if ax_xy:
        _style_ax(ax_xy, 'XY Bird\u2019s Eye (ENU)', 'East [m]', 'North [m]')
        ax_xy.set_aspect('equal', adjustable='box')
        ln_gt,  = ax_xy.plot([], [], color=_GT,  linewidth=1.6, label='Ground Truth', zorder=3)
        ln_est, = ax_xy.plot([], [], color=_EST, linewidth=1.3, linestyle='--', label='Predicted Trajectory', zorder=4)
        ln_pred, = ax_xy.plot([], [], color=_PRED, linewidth=2.0, linestyle='-.', label='IMM Prediction', zorder=5)
        mk_obs1 = ax_xy.scatter([], [], s=110, color=_OBS1, marker='^', zorder=6, edgecolors='white', linewidths=0.6)
        mk_obs2 = ax_xy.scatter([], [], s=110, color=_OBS2, marker='^', zorder=6, edgecolors='white', linewidths=0.6)
        mk_tgt  = ax_xy.scatter([], [], s=50, color=_GT,  marker='o', zorder=7, edgecolors='white', linewidths=0.8)
        mk_est  = ax_xy.scatter([], [], s=50, color=_EST, marker='D', zorder=7, edgecolors='white', linewidths=0.8, label='Predicted Target Position')
        ax_xy.legend(loc='upper left', fontsize=6.5, facecolor=_PANEL, edgecolor=_GRID, labelcolor=_TEXT, framealpha=0.9, ncol=2)

    # ── Setup Z altitude ──
    if ax_z:
        _style_ax(ax_z, 'Altitude Over Time (ENU Z\u2191)', 'Time [s]', 'Z [m]')
        ln_gt_z,  = ax_z.plot([], [], color=_GT,  linewidth=1.4, label='GT Z')
        ln_est_z, = ax_z.plot([], [], color=_EST, linewidth=1.1, linestyle='--', label='Estimate Z')
        ax_z.legend(loc='upper left', fontsize=7, facecolor=_PANEL, edgecolor=_GRID, labelcolor=_TEXT)

    # ── Setup Position Error ──
    if ax_err:
        _style_ax(ax_err, 'Position Error (3D)', 'Time [s]', 'Error [m]')
        ln_err, = ax_err.plot([], [], color=_ERR, linewidth=1.2, label='\u2016GT \u2212 Est\u2016')
        ln_avg, = ax_err.plot([], [], color=_ACCENT, linewidth=1.0, linestyle=':', label='Running Avg')
        ax_err.legend(loc='upper left', fontsize=7, facecolor=_PANEL, edgecolor=_GRID, labelcolor=_TEXT)

    # ── Setup Minimal 3D specific errors ──
    if ax_err_xy:
        _style_ax(ax_err_xy, 'Position Error (XY)', 'Time [s]', 'Error [m]')
        ln_err_xy, = ax_err_xy.plot([], [], color=_ERR, linewidth=1.2, label='\u2016GT \u2212 Est\u2016 XY')
        ln_avg_xy, = ax_err_xy.plot([], [], color=_ACCENT, linewidth=1.0, linestyle=':', label='Running Avg XY')
        ax_err_xy.legend(loc='upper left', fontsize=7, facecolor=_PANEL, edgecolor=_GRID, labelcolor=_TEXT)
        
    if ax_err_z:
        _style_ax(ax_err_z, 'Position Error (Z)', 'Time [s]', 'Error [m]')
        ln_err_z, = ax_err_z.plot([], [], color=_ERR, linewidth=1.2, label='|GT \u2212 Est| Z')
        ln_avg_z, = ax_err_z.plot([], [], color=_ACCENT, linewidth=1.0, linestyle=':', label='Running Avg Z')
        ax_err_z.legend(loc='upper left', fontsize=7, facecolor=_PANEL, edgecolor=_GRID, labelcolor=_TEXT)

    # ── Setup 3D view ──
    if ax_3d:
        _style_ax3d(ax_3d, '3D Trajectory vs GT')
        ln3_gt,  = ax_3d.plot([], [], [], color=_GT,  linewidth=1.6, label='Ground Truth', zorder=3)
        ln3_est, = ax_3d.plot([], [], [], color=_EST, linewidth=1.3, linestyle='--', label='Predicted Trajectory', zorder=4)
        ln3_pred, = ax_3d.plot([], [], [], color=_PRED, linewidth=2.0, linestyle='-.', label='IMM Prediction', zorder=5)
        mk3_obs1, = ax_3d.plot([], [], [], color=_OBS1, marker='^', markersize=10, markeredgecolor='white', markeredgewidth=0.6, linestyle='None', zorder=6)
        mk3_obs2, = ax_3d.plot([], [], [], color=_OBS2, marker='^', markersize=10, markeredgecolor='white', markeredgewidth=0.6, linestyle='None', zorder=6)
        mk3_tgt,  = ax_3d.plot([], [], [], color=_GT,  marker='o', markersize=6, markeredgecolor='white', markeredgewidth=0.8, linestyle='None', zorder=7)
        mk3_est,  = ax_3d.plot([], [], [], color=_EST, marker='D', markersize=6, markeredgecolor='white', markeredgewidth=0.8, linestyle='None', zorder=7, label='Predicted Target Position')
        ax_3d.legend(loc='upper left', fontsize=6.5, facecolor=_PANEL, edgecolor=_GRID, labelcolor=_TEXT, framealpha=0.9, ncol=2)
        
        # Add scroll zoom support
        zoom_factory(ax_3d)

    # ── Status bar ──
    ax_bar.set_facecolor(_BG)
    ax_bar.set_xlim(0, 1)
    ax_bar.set_ylim(0, 1)
    ax_bar.axis('off')
    status_txt = ax_bar.text(0.5, 0.5, '', transform=ax_bar.transAxes,
                             ha='center', va='center', fontsize=8,
                             fontfamily='monospace', color=_TEXT)

    def update(_frame):
        # Fetch thread-safe copies of data
        with node.lock:
            gt_data = list(node.gt_data)
            est_data = list(node.est_data)
            err_data = list(node.err_data)
            err_xy_data = list(node.err_xy_data)
            err_z_data = list(node.err_z_data)
            pred_xy = node.pred_xy.copy() if node.pred_xy is not None else None
            obs1_pos = node.obs1_pos.copy() if node.obs1_pos is not None else None
            obs2_pos = node.obs2_pos.copy() if node.obs2_pos is not None else None

        returns = []

        # --- XY panel ---
        if ax_xy:
            all_x, all_y = [], []
            if gt_data:
                arr = np.array([item[1] for item in gt_data])
                plot_arr = arr[::max(1, len(arr)//1000)]
                ln_gt.set_data(plot_arr[:, 1], plot_arr[:, 0])
                mk_tgt.set_offsets([[arr[-1, 1], arr[-1, 0]]])
                all_x.extend(arr[:, 1])
                all_y.extend(arr[:, 0])
            if est_data:
                arr = np.array([item[1] for item in est_data])
                plot_arr = arr[::max(1, len(arr)//1000)]
                ln_est.set_data(plot_arr[:, 1], plot_arr[:, 0])
                mk_est.set_offsets([[arr[-1, 1], arr[-1, 0]]])
                all_x.extend(arr[:, 1])
                all_y.extend(arr[:, 0])
            if pred_xy is not None and len(pred_xy) > 0:
                ln_pred.set_data(pred_xy[:, 1], pred_xy[:, 0])
            if obs1_pos is not None:
                mk_obs1.set_offsets([[obs1_pos[1], obs1_pos[0]]])
            if obs2_pos is not None:
                mk_obs2.set_offsets([[obs2_pos[1], obs2_pos[0]]])
            
            if all_x and all_y:
                if pred_xy is not None and len(pred_xy) > 0:
                    all_x.extend(pred_xy[:, 1])
                    all_y.extend(pred_xy[:, 0])
                if obs1_pos is not None:
                    all_x.append(obs1_pos[1])
                    all_y.append(obs1_pos[0])
                if obs2_pos is not None:
                    all_x.append(obs2_pos[1])
                    all_y.append(obs2_pos[0])

                min_x, max_x = min(all_x), max(all_x)
                min_y, max_y = min(all_y), max(all_y)
                cx, cy = (min_x + max_x) / 2.0, (min_y + max_y) / 2.0
                rx = max(max_x - min_x, 2.0)
                ry = max(max_y - min_y, 2.0)
                span = max(rx, ry) * 1.25
                ax_xy.set_xlim(cx - span / 2.0, cx + span / 2.0)
                ax_xy.set_ylim(cy - span / 2.0, cy + span / 2.0)
            else:
                ax_xy.relim()
                ax_xy.autoscale_view()

        # --- 3D panel ---
        if ax_3d:
            all_x3, all_y3, all_z3 = [], [], []
            if gt_data:
                arr = np.array([item[1] for item in gt_data])
                plot_arr = arr[::max(1, len(arr)//1000)]
                ln3_gt.set_data(plot_arr[:, 1], plot_arr[:, 0])
                ln3_gt.set_3d_properties(plot_arr[:, 2])
                mk3_tgt.set_data([arr[-1, 1]], [arr[-1, 0]])
                mk3_tgt.set_3d_properties([arr[-1, 2]])
                all_x3.extend(arr[:, 1])
                all_y3.extend(arr[:, 0])
                all_z3.extend(arr[:, 2])
            if est_data:
                arr = np.array([item[1] for item in est_data])
                plot_arr = arr[::max(1, len(arr)//1000)]
                ln3_est.set_data(plot_arr[:, 1], plot_arr[:, 0])
                ln3_est.set_3d_properties(plot_arr[:, 2])
                mk3_est.set_data([arr[-1, 1]], [arr[-1, 0]])
                mk3_est.set_3d_properties([arr[-1, 2]])
                all_x3.extend(arr[:, 1])
                all_y3.extend(arr[:, 0])
                all_z3.extend(arr[:, 2])
            if pred_xy is not None and len(pred_xy) > 0:
                ln3_pred.set_data(pred_xy[:, 1], pred_xy[:, 0])
                ln3_pred.set_3d_properties(pred_xy[:, 2])
                all_x3.extend(pred_xy[:, 1])
                all_y3.extend(pred_xy[:, 0])
                all_z3.extend(pred_xy[:, 2])
            if obs1_pos is not None:
                mk3_obs1.set_data([obs1_pos[1]], [obs1_pos[0]])
                mk3_obs1.set_3d_properties([obs1_pos[2]])
                all_x3.append(obs1_pos[1])
                all_y3.append(obs1_pos[0])
                all_z3.append(obs1_pos[2])
            if obs2_pos is not None:
                mk3_obs2.set_data([obs2_pos[1]], [obs2_pos[0]])
                mk3_obs2.set_3d_properties([obs2_pos[2]])
                all_x3.append(obs2_pos[1])
                all_y3.append(obs2_pos[0])
                all_z3.append(obs2_pos[2])
                
            if all_x3 and all_y3 and all_z3:
                # 1. Target the camera on the GT target (or estimate if GT missing)
                if gt_data:
                    ros_x, ros_y, ros_z = gt_data[-1][1]
                    tgt_cx, tgt_cy, tgt_cz = ros_y, ros_x, ros_z
                elif est_data:
                    ros_x, ros_y, ros_z = est_data[-1][1]
                    tgt_cx, tgt_cy, tgt_cz = ros_y, ros_x, ros_z
                else:
                    tgt_cx, tgt_cy, tgt_cz = 0.0, 0.0, 0.0
                
                # 2. Smooth camera tracking (adds a professional, cinematic feel)
                if not hasattr(ax_3d, '_cam_cx'):
                    ax_3d._cam_cx, ax_3d._cam_cy, ax_3d._cam_cz = tgt_cx, tgt_cy, tgt_cz
                else:
                    alpha = 0.3  # Smoothing factor
                    ax_3d._cam_cx += alpha * (tgt_cx - ax_3d._cam_cx)
                    ax_3d._cam_cy += alpha * (tgt_cy - ax_3d._cam_cy)
                    ax_3d._cam_cz += alpha * (tgt_cz - ax_3d._cam_cz)
                
                cx, cy, cz = ax_3d._cam_cx, ax_3d._cam_cy, ax_3d._cam_cz
                
                # 3. Use a fixed base span to prevent the camera from flying away
                # as the trajectory grows. The zoom factor adjusts this span.
                base_span = 30.0
                span = base_span * getattr(ax_3d, '_zoom_factor', 1.0)
                
                ax_3d.set_xlim3d(cx - span / 2.0, cx + span / 2.0)
                ax_3d.set_ylim3d(cy - span / 2.0, cy + span / 2.0)
                ax_3d.set_zlim3d(cz - span / 2.0, cz + span / 2.0)

        # --- Z panel ---
        if ax_z:
            if gt_data:
                t_gt = [item[0] for item in gt_data]
                z_gt = [item[1][2] for item in gt_data]
                ln_gt_z.set_data(t_gt, z_gt)
            if est_data:
                t_est = [item[0] for item in est_data]
                z_est = [item[1][2] for item in est_data]
                ln_est_z.set_data(t_est, z_est)
            ax_z.relim()
            ax_z.autoscale_view()

        # --- Error panel ---
        if ax_err:
            if err_data:
                t_err = [item[0] for item in err_data]
                errs = [item[1] for item in err_data]
                ln_err.set_data(t_err, errs)
                w = min(30, len(errs))
                avg = np.convolve(errs, np.ones(w) / w, mode='valid')
                ln_avg.set_data(t_err[w - 1:], avg)
                ax_err.relim()
                ax_err.autoscale_view()

        if ax_err_xy:
            if err_xy_data:
                t_err = [item[0] for item in err_xy_data]
                errs = [item[1] for item in err_xy_data]
                ln_err_xy.set_data(t_err, errs)
                w = min(30, len(errs))
                avg = np.convolve(errs, np.ones(w) / w, mode='valid')
                ln_avg_xy.set_data(t_err[w - 1:], avg)
                ax_err_xy.relim()
                ax_err_xy.autoscale_view()

        if ax_err_z:
            if err_z_data:
                t_err = [item[0] for item in err_z_data]
                errs = [item[1] for item in err_z_data]
                ln_err_z.set_data(t_err, errs)
                w = min(30, len(errs))
                avg = np.convolve(errs, np.ones(w) / w, mode='valid')
                ln_avg_z.set_data(t_err[w - 1:], avg)
                ax_err_z.relim()
                ax_err_z.autoscale_view()

        # --- Status bar ---
        elapsed = _time.monotonic() - node._t0
        with node.lock:
            cur_err = node.err_data[-1][1] if node.err_data else 0.0
            err_vals = [item[1] for item in node.err_data]
            gt_count = node._gt_count
            est_count = node._est_count
            gt_len = len(node.gt_data)
        
        avg_err = float(np.mean(err_vals)) if err_vals else 0.0
        gt_hz  = gt_count / max(elapsed, 1)
        est_hz = est_count / max(elapsed, 1)
        status_txt.set_text(
            f'GT: {gt_count} ({gt_hz:.1f} Hz)  │  '
            f'Est: {est_count} ({est_hz:.1f} Hz)  │  '
            f'Err: {cur_err:.3f} m  │  '
            f'Avg Err: {avg_err:.3f} m  │  '
            f'Buffer: {gt_len}/{MAX_TRAIL}'
        )

        return returns

    _ani = animation.FuncAnimation(fig, update, interval=UPDATE_INTERVAL_MS,
                                   blit=False, cache_frame_data=False)

    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
