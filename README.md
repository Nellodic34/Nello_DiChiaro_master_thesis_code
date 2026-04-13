# Vision Stack Deploy

Deployment package for the drone-hunters vision stack. It contains the ROS 2 launch scripts, the Python nodes for detection, triangulation, monitoring, evaluation, and the model weights used on the drone.

## What this repo does

The stack is split across two flying nodes and one monitoring node:

- `launch_observer_drone.sh` starts the observer drone pipeline.
- `launch_main_drone.sh` starts the main triangulation pipeline and, optionally, the evaluator.
- `launch_live_monitor.sh` starts a PC-side matplotlib monitor.

At runtime, the observer drone detects the target and publishes:

- the camera origin in world coordinates
- the detected ray in world coordinates

The main drone receives its own detections plus the remote ray from the observer drone, triangulates the target position, and then runs the IMM-based tracker to smooth and predict the target motion.

## Repository layout

- `config/runtime_stack.yaml`: central configuration for drone names, device selection, run mode, EKF/IMM tuning, and evaluation topic suffix.
- `nodes/yolov8_observer_node.py`: YOLOv8 detector on the observer drone.
- `nodes/yolov8_main_triangulation_node.py`: main detector + triangulation + tracking node.
- `nodes/evaluate_prediction_node.py`: trajectory evaluator used in `decentralized_evaluation` mode.
- `nodes/live_monitor.py`: local visualization of ground truth, estimate, prediction, and error.
- `launch_*.sh`: convenience launch scripts.
- `weights/`: local inference weights.

## How it works

### Observer drone

The observer node:

- subscribes to the observer camera image topic
- runs YOLO inference
- selects the best detection according to confidence and `target_class_id`
- computes a ray in the world frame using camera intrinsics and TF
- publishes:
  - `/<observer2>/camera_origin`
  - `/<observer2>/ray`
  - `/<observer2>/detection/compressed` when annotated output is enabled

### Main drone

The main node:

- subscribes to the main camera image topic
- runs YOLO inference locally
- computes the local ray for the best detection
- matches local and remote observations by timestamp
- triangulates the target position from the two rays
- feeds the measurement to an IMM filter with constant-velocity and constant-acceleration models
- publishes:
  - `/target/pose_estimate`
  - `/target/odometry_estimate`
  - `/target/position_estimate`
  - `/target/triangulation_raw`
  - `/target/predicted_trajectory`
  - marker topics for visualization

If no new measurement arrives, the main node can coast forward using the last filtered state.

### Evaluation and monitoring

When `launcher.drone_run_mode` is set to `decentralized_evaluation`, the main script also starts the evaluator node. The evaluator compares the predicted trajectory against ground truth and writes reports under `debug/evaluation/<timestamp>/`.

The live monitor is a separate PC-side tool that subscribes to the target odometry, estimate, prediction, and observer odometry topics and plots them in real time.

## Installation

Run the environment bootstrap script from this repo:

```bash
bash setup_env.sh
```

This creates a local `.venv` and installs the Python dependencies from `requirements.txt`.

## Running

### Observer drone

```bash
bash launch_observer_drone.sh
```

Optional argument:

```bash
bash launch_observer_drone.sh /path/to/model
```

### Main drone

```bash
bash launch_main_drone.sh
```

Optional argument:

```bash
bash launch_main_drone.sh /path/to/model
```

If `launcher.drone_run_mode` is `decentralized_evaluation`, this script also launches the evaluator node.

### Live monitor

```bash
bash launch_live_monitor.sh
```

## Default model selection

The default model path is now local to the repo:

- first choice: `weights/best_openvino_model`
- fallback: `weights/best.pt`

This applies to the triangulation and observer nodes, so the stack can run directly from the repo without pointing to a home-directory training path.

## Configuration: sim vs real drone

Most of the deployment differences are handled in `config/runtime_stack.yaml`.

### 1. Drone names

Update `uav_roles` to match your actual vehicle names:

- `observer1`: the main drone that runs the triangulation node
- `observer2`: the remote observer drone that publishes the ray
- `target`: the drone being tracked

These names are used to build the ROS topic prefixes.

### 2. Image topics

By default the nodes auto-discover the standard image topic names. If your setup uses a different camera topic, override:

- `vision.main_image_topic`
- `vision.observer_image_topic`

Use the compressed image topic if that is what your camera driver publishes, for example:

- `/uav1/rgb/image_raw/compressed`
- `/uav2/rgbd/color/image_raw/compressed`

### 3. Inference device

Set `vision.device` according to the platform:

- `cuda` for GPU-backed inference
- `cpu` for an Intel drone or any CPU-only deployment

### 4. Ground truth topic suffix

The evaluator and live monitor build the target odometry topic from:

- `/<target>/<evaluation.odom_topic_suffix>`

In simulation, the default is:

- `hw_api/ground_truth`

For the real drone, change it to the odometry topic suffix used by your flight stack, for example:

- `odometry/odom_main`

### 5. Annotated image publishing

For real deployment, keep this disabled unless you need debugging:

- `vision.publish_annotated_image: false`

Publishing annotated images increases bandwidth and can make the Wi-Fi link noisier.

### 6. Tracking and filter tuning

The `ekf` section controls the IMM and triangulation filter behavior:

- `kf_process_noise_acc`
- `kf_process_noise_vel`
- `imm_transition_prob`
- `kf_measurement_noise`
- `kf_adaptive_cov_max_multiplier`
- `kf_initial_covariance`
- `kf_mahalanobis_threshold`

The defaults are tuned for the current workflow, but you may need to adjust them if the camera baseline, target speed, or latency changes.

## Suggested setup for simulation

In simulation, you usually keep:

- `vision.device: cuda` if you have GPU acceleration
- `vision.publish_annotated_image: true` only if you want debug overlays
- `evaluation.odom_topic_suffix: hw_api/ground_truth`
- `launcher.drone_run_mode: decentralized_evaluation`

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
