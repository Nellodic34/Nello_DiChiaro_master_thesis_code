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

Legacy files such as `launch_main_drone.sh`, `nodes/yolov8_observer_node.py`, and `nodes/yolov8_main_triangulation_node.py` are not part of the new launch flow and can be removed after verification.

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

`config/runtime_stack.yaml` is the version copied from the current workspace. It now includes:

- launcher mode and `use_sim_time`
- observer-side ROI and image-space Kalman filter settings
- triangulation parameters exposed in YAML
- evaluation world-frame and odometry settings

Update at least these sections before real-drone use:

```yaml
uav_roles:
  observer1: uav1
  observer2: uav2
  target: uav3

vision:
  device: "cpu"
  main_image_topic: "..."
  observer_image_topic: "..."

evaluation:
  odom_topic_suffix: "odometry/odom_main"
```

## Model Paths

The copied runtime config keeps the workspace model path convention. The deploy package preserves compatibility aliases under `dataset_tools/weights` and `dataset_tools/weights2`, both pointing to the existing top-level weight directories, so the copied config continues to resolve correctly without duplicating weights.

## Notes

- The new launchers source `/opt/ros/...`, then optionally `~/ros2_ws/install/local_setup.bash`, then the local deploy `.venv`.
- Evaluation outputs stay relative to this deploy package, which is required for the drone-side setup.
- This migration does not modify the source workspace; only `/home/nello/vision_stack_deploy` is updated.
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
