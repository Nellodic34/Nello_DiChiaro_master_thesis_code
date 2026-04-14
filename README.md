# Vision Stack Deploy

Deployment package for the **drone-hunters** vision stack — a two-drone system that detects, triangulates, and predicts the trajectory of a target drone in real time, using only onboard cameras and CPU inference.

## Architecture overview

Two hunter drones fly simultaneously.  Each runs a YOLOv8/v11n detector on its camera feed.  The **observer drone** publishes the detected ray and camera origin in the world frame.  The **main drone** detects locally, receives the remote ray over Wi-Fi, triangulates the 3D target position from the two rays, and feeds it into an IMM (Interacting Multiple-Model) Kalman filter that estimates position, velocity and acceleration.  A predicted trajectory is published so that an interceptor can plan ahead.

```
Observer drone                          Main drone
┌──────────────────────┐       Wi-Fi    ┌──────────────────────────────────┐
│  Camera → YOLO → Ray ├──────────────► │  Camera → YOLO → Ray (local)     │
│  (ego_state)         │                │  + Remote ray  → Triangulation   │
└──────────────────────┘                │  → IMM Filter → Trajectory pred  │
                                        └──────────────────────────────────┘
```

## Repository layout

| Path | Purpose |
|------|---------|
| `config/runtime_stack.yaml` | Central configuration: drone names, device, ROI tuning, EKF/IMM params, evaluation |
| `nodes/yolov8_observer_node.py` | Observer drone: YOLO detection → ray + origin publishing |
| `nodes/yolov8_main_triangulation_node.py` | Main drone: YOLO detection + two-ray triangulation + IMM filter + trajectory prediction |
| `nodes/imm_image_overlay_node.py` | Overlay node: projects IMM predicted trajectory onto the observer camera image |
| `nodes/evaluate_prediction_node.py` | Trajectory evaluator: compares predictions to ground truth |
| `nodes/live_monitor.py` | PC-side matplotlib real-time visualization |
| `nodes/test_yolov8_realtime_node.py` | Minimal detection test node |
| `launch_observer_drone.sh` | Launch script for the observer pipeline |
| `launch_main_drone.sh` | Launch script for the main pipeline (+ evaluator in eval mode) |
| `launch_live_monitor.sh` | Launch script for the PC monitor |
| `weights/` | Model weights: OpenVINO IR, ONNX, PyTorch (.pt) |
| `debug/evaluation/` | Evaluation run outputs (JSON, CSV, markdown reports) |

## Key techniques

### 1. Adaptive ROI crop

After the first detection, YOLO inference switches from full-image (896 px) to a small crop (320 px) centered on the predicted target position.  This cuts inference time by ~70 % on CPU.

**ROI center prediction** uses two strategies in priority order:

- **IMM 3D→2D projection** — the main node projects the IMM-filtered 3D position through TF and camera intrinsics onto the image plane.  The observer node subscribes to `/target/odometry_estimate` and does the same projection, giving it access to the main node's IMM estimate.
- **2D pixel velocity tracker** — exponentially-smoothed pixel velocity from previous detections (fallback when IMM data is unavailable or the TF lookup fails).

**ROI escalation** prevents hard lock-on/lock-off behavior.  On consecutive misses the crop region gradually expands:

| Consecutive misses | ROI multiplier |
|--------------------|----------------|
| 0 | ×1.0 (base size) |
| 1–2 | ×1.5 |
| 3–5 | ×2.0 |
| ≥ 6 | full-image scan |

A detection at any stage resets the miss counter to zero.

**M-of-N init gate** blocks false positives from triggering the ROI lock.  When there is no active track, detections are buffered.  ROI tracking only starts when at least M of the last N detection centers cluster within a pixel spread threshold (defaults: 3-of-5 within 80 px).

**Prediction-gated detection selection** — when the ROI track is active, the best detection is chosen not by confidence alone but by a combined score `conf − λ × (d/d_max)`, where d is the pixel distance from the predicted ROI center and λ (default 0.3) controls the proximity bias.  This reduces the chance of jumping to a spurious high-confidence detection far from the target.

**Periodic full scan** — every N frames (default 10) a full-image scan is forced regardless of ROI state, as a safety net for reacquisition.

### 2. Two-ray triangulation

Each drone computes a 3D ray from its detection using camera intrinsics and the TF-derived camera-to-world transform.  The main node matches local and remote observations by timestamp (within `max_pair_age_sec`) and triangulates via closest-point-on-two-lines (midpoint formulation).

**Adaptive measurement covariance** — the triangulation error is anisotropic: it grows along the bisector of the two rays when the intersection angle is shallow.  The measurement covariance matrix R is constructed from the eigendecomposition of the ray geometry, with the depth variance scaled by `1/sin(angle)`, clamped at `kf_adaptive_cov_max_multiplier`.

**Kinematic ray shifting** — if the local and remote observations have a non-zero timestamp gap (asynchronous cameras or Wi-Fi jitter), the remote ray origin is shifted by `velocity × Δt` using the current IMM velocity estimate, effectively projecting it to the local timestamp before triangulation.

### 3. IMM filter (CV + CA)

An Interacting Multiple-Model filter runs two parallel Kalman filters:

- **Constant Velocity (CV)** — state `[x, y, z, vx, vy, vz]`, 6-D
- **Constant Acceleration (CA)** — state `[x, y, z, vx, vy, vz, ax, ay, az]`, 9-D

At each measurement step the filters are mixed, predicted, corrected, and re-weighted by their measurement likelihood.  The combined output (position, velocity, acceleration, covariance) is published as an `Odometry` message.

**Mahalanobis gating** rejects outlier measurements that are statistically too far from both filter predictions, causing the filter to coast on its own kinematic model instead of incorporating a corrupt measurement.

**Covariance reset** — if the covariance trace exceeds a threshold, the filter is re-initialized at the current measurement to recover from divergence.

**Coast prediction** — when no new triangulation arrives (e.g. Wi-Fi dropout), the filter extrapolates forward using the last kinematic state.

### 4. OpenVINO inference

For Intel-CPU drones the model is exported to OpenVINO IR format.  The default model path is `weights/best_openvino_model` (fixed input shape 896).  Fallback is `weights/best.pt` (PyTorch).  On GPU-equipped machines, set `vision.device: cuda`.

### 5. Annotated image publishing

When `vision.publish_annotated_image` is `true`, each node publishes `/<uav>/detection/compressed` with:

- Bounding boxes for all detections (green) with the best detection highlighted (orange, marked `*`)
- A **green ROI rectangle** showing the exact crop region used for inference, labeled "ROI"

Keep this disabled in production to save Wi-Fi bandwidth.

## Installation

```bash
bash setup_env.sh
```

Creates a local `.venv` and installs Python dependencies from `requirements.txt` (including `numpy<2` for cv_bridge compatibility).

## Running

### Observer drone

```bash
bash launch_observer_drone.sh [/path/to/model]
```

### Main drone

```bash
bash launch_main_drone.sh [/path/to/model]
```

In `decentralized_evaluation` mode this also starts the evaluator node.

### Live monitor (PC side)

```bash
bash launch_live_monitor.sh
```

## Configuration — `config/runtime_stack.yaml`

### Drone names

```yaml
uav_roles:
  observer1: uav1   # main drone (runs triangulation)
  observer2: uav2   # remote observer (publishes ray)
  target: uav3      # drone being tracked
```

### Image topics

Leave empty for auto-discovery, or override:

```yaml
vision:
  main_image_topic: "/uav1/rgbd/color/image_raw/compressed"
  observer_image_topic: "/uav2/rgbd/color/image_raw/compressed"
```

### Inference device

```yaml
vision:
  device: "cpu"    # "cpu" for Intel drone, "cuda" for GPU
```

### ROI crop tuning

```yaml
vision:
  roi_crop_enabled: true
  roi_crop_imgsz: 320
  roi_margin_multiplier: 3.0
  roi_min_size_px: 200
  roi_track_timeout_sec: 0.5
  roi_full_scan_interval: 10
  roi_init_m: 3
  roi_init_n: 5
  roi_init_max_spread_px: 80
  roi_selection_lambda: 0.3
```

### IMM filter tuning

```yaml
ekf:
  kf_process_noise_acc: 1.25
  kf_process_noise_vel: 1.2
  imm_transition_prob: 0.90
  kf_measurement_noise: 0.05
  kf_adaptive_cov_max_multiplier: 50.0
  kf_initial_covariance: 2.0
  kf_mahalanobis_threshold: 7.0
```

### Ground truth / evaluation

```yaml
evaluation:
  odom_topic_suffix: "hw_api/ground_truth"   # simulation
  # odom_topic_suffix: "odometry/odom_main"  # real drone
```

### Sim vs real quick reference

| Setting | Simulation | Real drone |
|---------|-----------|------------|
| `vision.device` | `cuda` | `cpu` |
| `vision.publish_annotated_image` | `true` | `false` (save bandwidth) |
| `evaluation.odom_topic_suffix` | `hw_api/ground_truth` | `odometry/odom_main` |
| `launcher.drone_run_mode` | `decentralized_evaluation` | `decentralized` |

## Default model selection

Both nodes auto-select the model in this order:

1. `weights/best_openvino_model` (OpenVINO IR, fastest on Intel CPU)
2. `weights/best.pt` (PyTorch fallback)

Override via the launch script argument or the `model_path` ROS parameter.

That gives you tracking plus automatic evaluation outputs.

## Suggested setup for real drones

For real hardware, the usual changes are:

- map `uav_roles` to the actual drone names
- set the correct camera topics if they differ from the defaults
- set `vision.device: cpu` if the onboard computer is CPU-only
- keep `vision.publish_annotated_image: false`
- change `evaluation.odom_topic_suffix` to the real odometry topic suffix

The main and observer launch scripts will then use the same repo layout and the local `weights/` folder on the drone.

## Notes

- The launch scripts try a local `.venv` first.
- If that is missing, they fall back to a workspace venv path defined in the script.
- The repo is designed to be run from the workspace root, with weights stored in `weights/`.

## Troubleshooting

- If a node says the model cannot be found, verify that `weights/best_openvino_model` or `weights/best.pt` exists.
- If TF lookup fails, check that the camera frame and world frame are published correctly.
- If the live monitor shows no data, confirm that the target odometry suffix in `runtime_stack.yaml` matches the running platform.
- If ROS is not found, make sure `/opt/ros/<distro>/setup.bash` exists and is sourced correctly.
