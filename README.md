# Vision Stack Deploy

Two drones (**observers**) watch a third one (the **target**) and estimate where it is,
using only their cameras. Each observer detects the target with YOLO and turns the
detection into a **ray** (a direction in space). A third process crosses the two rays,
filters the result, and publishes the target's position and velocity on
**`/target/odometry_estimate`**.

> **Where the results go.** Every run writes its evaluation to
> **`debug/evaluation/<timestamp>/`**. `debug/evaluation/latest` always points to the
> newest one. In the folder you'll find the plots (`prediction_eval_*.png`: trajectory,
> XY, XYZ, distance, estimate vs. odometry), a markdown report
> (`prediction_eval_report.md`: ADE, FDE, RMSE, error split between STEREO and COAST), the raw numbers
> (`prediction_eval.json`) and a per-sample CSV. On a real flight they end up on **uav11**,
> which runs the evaluator. See [Results](#results).

---

## How it works

```
 uav11 camera ──► observer_node (observer1) ──► /uav11/target_observation ─┐
                                                                          ├─► triangulation_node ──► /target/odometry_estimate
 uav8 camera  ──► observer_node (observer2) ──► /uav8/target_observation  ─┘     (runs on uav11)           │
                                                                                                           ├─► live_monitor (ground PC)
                                                                                                           └─► evaluator ──► debug/evaluation/
```

**1. Observer (`nodes/observer_node.py`, one per drone).**
- It reads its camera image topic (`vision.main_image_topic` for observer1,
  `vision.observer_image_topic` for observer2), the camera info, and the drone's own RTK pose.
- YOLO (`vision.model_path`) finds the target in the image. Once the target has been found,
  it searches only a crop around it.
- Detections are filtered. Anything that is really the *other* observer drone gets dropped.
  A 2D Kalman filter smooths the box and only trusts it after 3 detections in a row.
- The box centre becomes a ray in the world frame: it starts at the camera and points at the
  target. The ray is stamped with the time the image was captured.
- The ray is packed (timestamp, origin, direction, confidence) and published on
  `/<uav>/target_observation`.

**2. Triangulation (`nodes/triangulation_node.py`, on uav11).**
- It pairs one ray from each drone whose timestamps are close (≤ `triangulation.max_pair_age_sec`).
- It computes the point where the two rays come closest.
- An IMM filter (a constant-velocity and a constant-acceleration model, blended) turns those
  points into a smooth position + velocity, plus a short predicted trajectory.
- If no pair arrives for a moment, it keeps predicting on its own ("COAST").

**3. Ground side.**
- `nodes/live_monitor.py` plots estimate, ground truth, prediction and both observers live.
- `nodes/evaluate_prediction_node.py` compares the estimate with the target's own RTK
  position and writes the report above.

Depth is **not** used: range comes only from the geometry of two rays crossing. That is why
what matters most is pointing accuracy (camera attitude) and clock sync between the drones.

### Useful topics

| Topic | Published by | What it is |
|-------|--------------|------------|
| `/target/odometry_estimate` | triangulation | **Final result**: position, velocity, covariance |
| `/target/pose_estimate`, `/target/position_estimate` | triangulation | Same result, lighter message types |
| `/target/predicted_trajectory` | triangulation | Where the target is expected to go next (`Path`) |
| `/target/triangulation_raw` | triangulation | The raw crossing point, before filtering |
| `/target/ground_truth_aligned` | triangulation | Target's RTK position in the same frame as the estimate |
| `/target/position_error` | triangulation | Estimate minus ground truth |
| `/tracker/fusion_mode` | triangulation | `STEREO` (two rays), `SINGLE_DEPTH` (one ray + depth, off today), `COAST` (predicting) |
| `/target/position_markers`, `/target/predicted_trajectory_markers` | triangulation | RViz markers |
| `/uav11/triangulation_debug_annotated/compressed` | triangulation | uav11 image with both rays drawn on it |
| `/<uav>/target_observation` | observer | The ray packet (binary) |
| `/<uav>/detection2/compressed` | observer | Camera image with the detection box, the best way to see what a drone sees |
| `/<uav>/estimation_manager/odom_main` | drone autopilot | Input: each drone's own RTK pose |

For a longer walkthrough, open [`docs/architecture.html`](docs/architecture.html) in a browser.

---

## Quick start

Three machines, four processes, all started by hand:

| Machine | What | Command |
|---------|------|---------|
| uav8 (observer2) | observer | `bash launch_observer_detection_node.sh` |
| uav11 (observer1) | observer | `bash launch_main_detection_node.sh` |
| uav11, second terminal | triangulation (+ evaluator) | `bash launch_triangulation_node.sh` |
| Ground PC | live monitor | `bash launch_live_monitor.sh` |

### 1. Install (once per machine)

```bash
git clone <this repo> && cd vision_stack_deploy
bash setup_env.sh               # creates .venv with requirements.txt
git rev-parse --short HEAD      # must print the SAME hash on all three machines
```

You need ROS 2 Jazzy in `/opt/ros/jazzy`. The launchers source it and `.venv` for you.

### 2. Check the config

Every setting lives in **`config/runtime_stack.yaml`**. If this README and the YAML
disagree, the YAML wins. Before each flight, check these:

| Setting | Should be |
|---------|-----------|
| `uav_roles.observer1` / `observer2` / `target` | the real drone names (today `uav11` / `uav8` / `uav2`); the image topics under `vision:` also contain the names |
| `vision.model_path` | the weights you really mean to fly (today `weights8`) |
| `launcher.use_sim_time` | `false` (real drones publish no `/clock`) |
| `launcher.drone_run_mode` | `decentralized_evaluation` if you want the report, `decentralized` otherwise |
| `bag_profile` | `"none"` (anything else is a bias fitted on an old flight) |
| `shared_heading` | `false` (bench-only feature) |
| `evaluation.shared_world_origin` | `true` only if the Z check in step 3 passes |

The pixel settings under `kalman_filter:` are tuned for 1920×1080. Each line's comment gives
the values for 1280×720 and 640×480. What every value means and why it was chosen:
[`docs/config_rationale.md`](docs/config_rationale.md).

### 3. Preflight (both drones on the same flat ground, RTK fixed)

```bash
python3 preflight_check.py          # config check; "NOT READY TO FLY" = stop
python3 preflight_check.py --live   # attitude check; differential tilt > 1° = reboot that FCU
chronyc tracking                    # clocks: offset must be well under 50 ms
```

Z datum check. Run it **on each drone over ssh**, because `/tf_static` does not cross the
network:

```bash
ros2 run tf2_ros tf2_echo uav11/utm_origin uav11/world_origin
ros2 run tf2_ros tf2_echo uav8/utm_origin uav8/world_origin
```

If the two Z values are within ~0.30 m, `shared_world_origin: true` is fine. If they are
not, do not fly with two drones: the rays from uav8 would be biased or discarded.

### 4. Start, in this order

1. uav8 observer. Wait for `Observer node: FPS=...` (about 10 FPS).
2. uav11 observer. Its FPS should be similar to uav8's.
3. uav11 triangulation. Wait for `Triangulation node initialization complete.`
4. Ground monitor.

### 5. Ground check, before takeoff

With the target visible from **both** cameras:

```bash
ros2 topic echo /tracker/fusion_mode         # must reach STEREO
ros2 topic hz /target/odometry_estimate      # ~10 Hz
```

If it never reaches `STEREO` on the ground, it won't in the air. If you record a bag, start
it before takeoff and check it is growing (`du -sh <bag_dir>`).

### 6. Stop

Ctrl-C the triangulation terminal and **wait**: the evaluator needs up to a minute to write
the plots. Then stop the observers and the monitor.

---

## Results

With `drone_run_mode: decentralized_evaluation`, the triangulation launcher also starts
`evaluate_prediction_node.py`. The evaluator listens *during* the flight and writes
everything when you stop it, into `debug/evaluation/<timestamp>/` on uav11:

| File | Content |
|------|---------|
| `prediction_eval_report.md` | Summary: ADE, FDE, RMSE position/velocity, max error, ANEES, RMSE split STEREO vs. COAST |
| `prediction_eval_plot.png`, `_xy.png`, `_xyz.png` | Estimate and prediction vs. ground truth |
| `prediction_eval_distance.png` | Error vs. distance from the observers |
| `prediction_eval_est_vs_odom.png` | Estimate vs. the target's own odometry |
| `prediction_eval.json`, `prediction_eval_samples.csv` | Raw numbers |

`bash run_prediction_evaluation.sh` starts the evaluator on its own, for example on the
ground PC or next to a bag replay. Like the automatic one, it has to run **while** the data
is flowing. Run it after the flight and there is nothing left to listen to.
`AUTO_OPEN_RESULTS=1` opens the files when it finishes.

---

## Offline replay (work on the filter without flying)

```bash
./replay_bag_a.sh                  # play the Temesvar bag, show monitor, score on Ctrl-C
./replay_bag_a.sh --no-images      # without the two image windows
KF_PROCESS_NOISE_VEL=8.0 ./replay_bag_a.sh --no-images   # try a different value
```

It runs the triangulation node against a recorded flight on its own `ROS_DOMAIN_ID` (45),
so it won't disturb a live session. When you stop it, it prints a score, which is also saved
to `~/replay_runs/<time>/score.txt`. Add `DRONE_RUN_MODE=decentralized_evaluation` in front
of the command to also get the full `debug/evaluation/` report. Every launcher lets
environment variables override the YAML; the names are at the top of each script.

Self-checks:

```bash
python3 -m py_compile nodes/*.py
python3 nodes/test_coast_singer.py
python3 nodes/test_live_monitor_gt_fallback.py
python3 nodes/test_debug_overlay_sides.py
```

---

## Repository layout

| Path | What |
|------|------|
| `config/runtime_stack.yaml` | All runtime settings |
| `nodes/observer_node.py` | Observer: image → detection → ray |
| `nodes/triangulation_node.py` | Two rays → point → IMM filter → `/target/odometry_estimate` |
| `nodes/evaluate_prediction_node.py` | Evaluator → `debug/evaluation/` |
| `nodes/live_monitor.py` | Ground-station live plots |
| `nodes/test_*.py` | Self-checks (`test_yolov8_realtime_node.py` is the detection-only mode, not a test) |
| `launch_*.sh` | One launcher per process (table above) |
| `run_prediction_evaluation.sh` | Evaluator on its own |
| `replay_bag_a.sh`, `dataset_tools/score_replay.py` | Offline replay and scoring |
| `preflight_check.py` | Preflight gate |
| `PREFLIGHT_CHECKLIST.txt` | Printable field checklist (Italian) |
| `docs/architecture.html` | Detailed architecture walkthrough |
| `docs/config_rationale.md` | Why every setting has its value |
| `dataset_tools/weights*/` | YOLO weights |

---

## Troubleshooting

| Symptom | Likely cause |
|---------|--------------|
| `fusion_mode` stuck on `COAST` | (a) uav8's rays are discarded: check the Z datum and `shared_world_origin`. (b) Rays are never published: a ray needs 3 detections **in a row**, so if detections flicker, you can see a box on `/uav8/detection2` and still get no ray. |
| Nodes start but do nothing | `use_sim_time: true` on real drones |
| TF lookup fails / no rays | `vision.publish_camera_static_tf` must be `true` on the drones |
| Live monitor empty | `evaluation.odom_topic_suffix` must be `estimation_manager/odom_main` on real drones |
| Estimate jumps between two tracks | a node from a previous run is still publishing; kill it |
| Model not found | `vision.model_path` is relative to the repo root |
| `/uav8/triangulation_debug_annotated` always empty | expected: uav11 never receives uav8's camera frames. Use `/uav8/detection2/compressed` |
| uav8 re-acquires the target slower than uav11 | known and deferred; not a field problem |
| No `debug/evaluation/` folder after the flight | `drone_run_mode` was not `decentralized_evaluation`, or the triangulation was killed before the evaluator finished writing |
