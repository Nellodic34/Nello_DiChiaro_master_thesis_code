# Vision Stack Deploy

Deployment package for the current two-drone vision stack used on real hardware.

This package now uses the split pipeline from the current workspace:

- `nodes/observer_node.py` for per-drone detection and frozen bearing publication
- `nodes/triangulation_node.py` for two-observer matching, triangulation, and IMM filtering
- `nodes/evaluate_prediction_node.py` for trajectory evaluation into `debug/evaluation`
- `nodes/live_monitor.py` for the PC-side monitor

## Repository Layout

| Path | Purpose |
|------|---------|
| `config/runtime_stack.yaml` | Runtime configuration copied from the current workspace |
| `nodes/observer_node.py` | Detection node used on both drones with role-based config |
| `nodes/triangulation_node.py` | Triangulation and IMM fusion node |
| `nodes/evaluate_prediction_node.py` | Prediction evaluator with relative output under `debug/evaluation` |
| `nodes/live_monitor.py` | Live monitor for a PC on the same ROS graph |
| `nodes/test_yolov8_realtime_node.py` | Simple local detection test path used by `simple_detection` mode |
| `launch_observer_detection_node.sh` | Start detection on the observer drone (`observer2`) |
| `launch_main_detection_node.sh` | Start detection on the main drone (`observer1`) |
| `launch_triangulation_node.sh` | Start triangulation and IMM fusion on the main drone |
| `launch_live_monitor.sh` | Start the PC live monitor |
| `run_prediction_evaluation.sh` | Run the evaluator and write results under `debug/evaluation/<timestamp>` |
| `weights/`, `weights2/` | Existing deploy weights kept unchanged |

Legacy files `launch_main_drone.sh`, `nodes/yolov8_observer_node.py`, and `nodes/yolov8_main_triangulation_node.py` have been removed — they were not part of the launch flow above and are superseded by `observer_node.py` + `triangulation_node.py`.

## Installation

```bash
bash setup_env.sh
```

`setup_env.sh` and `requirements.txt` were intentionally left unchanged. The replaced nodes keep the same non-ROS Python dependency profile already used by the previous deploy package.

## Launch Flow

### Observer Drone

```bash
bash launch_observer_detection_node.sh
```

This launches `nodes/observer_node.py` with role `observer2`.

### Main Drone

Run these in separate terminals on the main drone:

```bash
bash launch_main_detection_node.sh
bash launch_triangulation_node.sh
```

The first launches `nodes/observer_node.py` with role `observer1`. The second launches `nodes/triangulation_node.py`.

### Live Monitor

```bash
bash launch_live_monitor.sh
```

### Evaluation

```bash
bash run_prediction_evaluation.sh
```

Artifacts are written to `debug/evaluation/<timestamp>` and `debug/evaluation/latest` is updated to point to the most recent run.

## Runtime Configuration

`config/runtime_stack.yaml` was resynced from the current workspace (2026-07-21). It now includes:

- launcher mode and `use_sim_time`
- observer-side ROI and image-space Kalman filter settings
- triangulation parameters, IMM Markov tuning, and the CRLB normalization switch (`ekf.normalize_crlb`)
- prediction-guided ray pairing (`triangulation.pairing_*`)
- optional feature blocks, all disabled by default: `tight_coupling` (3D→2D feedback gate), `depth_fusion` (RGB-D range fusion), `observer_odom_uncertainty` (WLS/CRLB propagation), `observer_self_rejection` (intra-swarm false-positive rejection)
- evaluation world-frame/odometry settings and `evaluation.live_monitor_mode`

Real-drone production values already set in this file (double-check they still match your hardware before flying):

```yaml
uav_roles:
  observer1: uav11
  observer2: uav8
  target: uav28

vision:
  use_sim_time: false
  device: "cpu"
  publish_annotated_image: false
  publish_detector_evaluation_packets: true
  main_image_topic: "/uav11/rgbd/color/image_raw/compressed"
  observer_image_topic: "/uav8/rgbd/color/image_raw/compressed"
  model_path: "dataset_tools/weights2/best_openvino_model"
```

**EKF/CRLB tuning caveat**: `ekf.*` and `kalman_filter.*` were carried over from the workspace's latest CMA-ES/CRLB retuning pass (`normalize_crlb: false`, `kf_mahalanobis_threshold: 34.0`, new IMM Markov probabilities, `kf_measurement_noise: 0.0002`, etc.). The workspace's own commit history flags these values as **not yet fully validated on real hardware** — treat the first flights after this sync as a validation run for the filter, not a routine deploy.

**Known unresolved placeholder**: `evaluation.odom_topic_suffix` is still `"estimation_manager/odom_main"`, which the file's own inline comment labels as the *simulation* ground-truth topic. This was already the case before this sync and was left untouched because the correct real-drone topic suffix wasn't confirmed — change it to the real odometry topic (e.g. `odometry/odom_main`) before trusting `run_prediction_evaluation.sh` / the live monitor on real flights.

## Model Paths

The runtime config keeps the workspace model path convention. The deploy package preserves compatibility aliases under `dataset_tools/weights` and `dataset_tools/weights2`, both pointing to the existing top-level weight directories, so the config continues to resolve correctly without duplicating weights.

## Notes

- The launchers source `/opt/ros/...`, then optionally `~/ros2_ws/install/local_setup.bash`, then the local deploy `.venv`.
- Evaluation outputs stay relative to this deploy package, which is required for the drone-side setup.
- This migration does not modify the source workspace; only `/home/nello/vision_stack_deploy` is updated.
- The main and observer launch scripts use the same repo layout and the local `weights/` folder on the drone.

## Troubleshooting

- If a node says the model cannot be found, verify that `weights/best_openvino_model` or `weights/best.pt` exists.
- If TF lookup fails, check that the camera frame and world frame are published correctly.
- If the live monitor shows no data, confirm that the target odometry suffix in `runtime_stack.yaml` matches the running platform.
- If ROS is not found, make sure `/opt/ros/<distro>/setup.bash` exists and is sourced correctly.
