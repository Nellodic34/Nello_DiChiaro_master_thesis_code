# sim/ — running this stack in simulation

This folder runs **the nodes of this repo** (`nodes/observer_node.py`,
`nodes/triangulation_node.py`, the live monitor and the evaluator) inside the
FlightForge simulator. Nothing in it goes on the drones. It does not edit any
tracked file either, so the config you test in sim is the config that flies.

The simulator side (FlightForge, the MRS UAV System, the tmux session that spawns
the drones) lives in a second repo, **training_stack**. This folder only swaps the
perception panes of that session for the nodes of this repo.

```
 training_stack (sim tree)                      vision_stack_deploy (this repo)
 ─────────────────────────                      ───────────────────────────────
 FlightForge + MRS, spawns uav1/uav2/uav3  ──►  sim/observer1.sh  ─► nodes/observer_node.py (uav1)
 launch_sim.sh opens the tmux panes             sim/observer2.sh  ─► nodes/observer_node.py (uav2)
                                                sim/triangulation.sh ─► nodes/triangulation_node.py
                                                sim/live_monitor.sh  ─► nodes/live_monitor.py
                                                sim/evaluation.sh    ─► nodes/evaluate_prediction_node.py
```

---

## 1. What you need

- Ubuntu 24.04, ROS 2 Jazzy (`/opt/ros/jazzy`).
- An **NVIDIA GPU with a working driver** (`nvidia-smi` must work). FlightForge
  refuses to start on the Intel/Mesa renderer unless you force it (see
  [Troubleshooting](#6-troubleshooting)).
- `gnome-terminal`, `tmux`, `tmuxinator` (`sudo apt install gnome-terminal tmux tmuxinator`).
- The **MRS UAV System for ROS 2**:
  <https://github.com/ctu-mrs/mrs_uav_system/tree/ros2>. The launcher sources
  `~/ros2_ws/install/local_setup.bash` if that file exists. Otherwise it sources the
  `local_setup.bash` of the training_stack repo.
- The **FlightForge simulator binary**. This is the Unreal build, not the ROS package.
  By default the launcher looks for it in `~/Pliska_FlightForge/` and starts it with
  `./mrs_flight_forge.sh`. The ROS package and its docs are here:
  <https://github.com/ctu-mrs/mrs_uav_flightforge_simulator/tree/ros2>.
  <!-- TODO: add the exact download link of the FlightForge build we use -->

## 2. One-time setup

**Clone both repos exactly at these paths.** `sim/` looks for the sim tree in
`~/git/training`. The default `git clone` would create `training_stack/`, so give it
the target directory explicitly:

    git clone -b master https://mrs.fel.cvut.cz/gitlab/drone-hunters/vision_stack_deploy.git ~/vision_stack_deploy
    git clone -b dev https://mrs.fel.cvut.cz/gitlab/drone-hunters/training_stack.git ~/git/training

(A different location also works if you export `SIM_TREE=<path>/mrs_uav_flightforge_simulator/tmux/one_drone`.)

**Two virtualenvs, one per repo.** Each repo has its own dependencies:

    # sim tree: runs launch_sim.sh, the config generator and the stereo depth node
    python3.12 -m venv --system-site-packages ~/git/training/mrs_uav_flightforge_simulator/.venv
    ~/git/training/mrs_uav_flightforge_simulator/.venv/bin/pip install -r ~/git/training/requirements.txt

    # this repo: runs the perception nodes
    ~/vision_stack_deploy/setup_env.sh

`--system-site-packages` is required on the first one, because that venv has to see
`rclpy`, `cv_bridge` and the other system ROS packages.

**FlightForge somewhere else?** Export the path before launching (or add it to your `~/.bashrc`):

    export FLIGHTFORGE_DIR=/path/to/your/FlightForge     # default: ~/Pliska_FlightForge
    export FLIGHTFORGE_CMD=./mrs_flight_forge.sh         # default, relative to FLIGHTFORGE_DIR

**`launch_sim.sh` patch.** This is already committed on the `dev` branch of
training_stack, so normally you skip it. `run_sim.sh` checks for the patch and, if it
is missing, prints the one command that applies it:

    patch -d ~/git/training/mrs_uav_flightforge_simulator/tmux/one_drone launch_sim.sh < ~/vision_stack_deploy/sim/launch_sim.patch

## 3. Running

    ~/vision_stack_deploy/sim/run_sim.sh

**Use this launcher and not the `launch_*.sh` scripts of the training tree.** Those
start the training tree's own copies of the nodes, which read the training tree's yaml.

What happens:

1. A new terminal opens with the tmux session `simulation` (socket `mrs`):
   FlightForge, the MRS stack, and the three drones uav1, uav2, uav3.
2. A tmux window `triangulation` holds three panes: observer uav1, observer uav2,
   triangulation. All three run the nodes of this repo.
3. A tmux window `evaluation` holds the evaluator and the live monitor.
4. To make the target move, use the utilities in the sim tree
   (`~/git/training/mrs_uav_flightforge_simulator/tmux/one_drone/`: `takeoff.sh`,
   `goto.py`, `path.py`).

To stop everything: `~/git/training/mrs_uav_flightforge_simulator/tmux/one_drone/kill.sh`.

Results are written to `~/vision_stack_deploy/debug/evaluation/<timestamp>/`, the same
place a real flight writes to (see the main [README](../README.md#results)).

**Quick checks once it is running:**

    ros2 topic hz /uav1/target_observation      # rays from observer1
    ros2 topic hz /uav2/target_observation      # rays from observer2
    ros2 topic echo /target/odometry_estimate --once

## 4. Where to change what

Two yaml files, two separate jobs. They do not overlap.

| What you want to change | File |
|---|---|
| **Vision, triangulation, KF/IMM, gating, ROI, every perception knob** | `~/vision_stack_deploy/config/runtime_stack.yaml` (**this one always wins** for the nodes of this repo) |
| The world: how many drones, their names (`uav_roles`), simulator config, whether the evaluator pane opens (`evaluation.auto_start_with_launch_sim`) | `~/git/training/mrs_uav_flightforge_simulator/tmux/one_drone/config/runtime_stack.yaml` |
| GPS origin of the world (drones crash right after a command = wrong origin) | `.../one_drone/config/world_config.yaml` |

The nodes find their config on their own, as `<repo>/config/runtime_stack.yaml`
relative to the node file. There is no environment variable that points them at
another yaml. Because `run_sim.sh` runs the nodes of *this* repo, they always read
*this* repo's config. That is the reason `sim/` exists at all.

The drone names come from `uav_roles` in the **sim** yaml, so if you change the world
there, you do not have to touch anything here.

**What the harness forces on purpose.** These values do NOT come from the deploy
yaml, because the sim needs something different from the real drones:

| Setting | In sim | Why | Override |
|---|---|---|---|
| `use_sim_time` | `true` | the sim publishes `/clock`; real drones do not | — |
| `drone_run_mode` | `decentralized` | the evaluator already runs in its own pane; `decentralized_evaluation` would start a second one | `DRONE_RUN_MODE=...` |
| `bag_profile` | `none` | the per-flight bias constants describe a real bag, not the sim | — |
| `model_path` | `weights_sim` of the sim tree | `weights8` is trained on real footage and may detect nothing on synthetic images | `MODEL_PATH=/abs/path/best_openvino_model` |
| `device` | `cuda` | sim runs on a GPU PC | edit `observer1.sh` / `observer2.sh` |
| `publish_camera_static_tf` | `false` | the sim already publishes the camera TF chain | — |

Any extra argument to `run_sim.sh` is passed through to `launch_sim.sh`.

Other environment variables it honours: `SIM_TREE`, `FLIGHTFORGE_DIR`,
`FLIGHTFORGE_CMD`, `ROS_SETUP`, `ROS2_WS_LOCAL_SETUP`, `VENV_ACTIVATE_SIM` (venv of
this repo, if it is not in `~/vision_stack_deploy/.venv`), `ALLOW_INTEL_RENDERER=1`.

**The training tree also has copies of the nodes.** They are in
`one_drone/dataset_tools/` and kept identical to `nodes/` here. `run_sim.sh` does not
use them. The CMA-ES tuner (`one_drone/tuning/`) imports its filter classes from them.
If you change `nodes/observer_node.py` or `nodes/triangulation_node.py`, copy the
file over there as well, or the tuner optimises a filter that is not the one that flies.

## 5. How the harness works (the four traps)

The deploy nodes are written for the real drones (uav11/uav8, RealSense topics
`/uavX/rgbd/color/...`). The sim has uav1/uav2/uav3 and `/uavX/rgb/image_raw`.
`common.sh` reads **both** yaml files and maps one onto the other:

| What | How it is redirected |
|---|---|
| drone names, roles, sim time, KF knobs | environment variables the launchers already read |
| `model_path`, `device`, `publish_camera_static_tf` | ROS parameters, passed via `"$@"` |
| image and camera_info topics | **ROS remaps only**. They are not parameters; the node reads them from the yaml |

Things that are easy to get wrong:

1. **Message type.** `observer_node` chooses `CompressedImage` or `Image` from the
   `/compressed` suffix of the *configured* topic name, not of the remapped one. So
   after the remap it still wants compressed images, while FlightForge publishes raw.
   `ensure_compressed` starts an `image_transport republish` only if the compressed
   topic does not already exist.
2. **Camera TF.** On the real drone the node publishes the fcu->camera chain itself,
   with the real mount offsets. In sim that chain already exists, so the shims pass
   `publish_camera_static_tf:=false` and the frame comes from the simulator's `CameraInfo`.
3. **Weights.** See `model_path` above.
4. **Images are not the only thing to remap.** `live_monitor.py` declares no ROS
   parameters. It builds the observer odometry topics as
   `/{uav_roles observer}/{odom_topic_suffix}` from the deploy yaml. In sim it would
   wait on `/uav11` and `/uav8`, and the observer triangles would be missing from the
   plot. Every topic that is built from a drone name has to be remapped.

## 6. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `launch_sim.sh non e' ancora patchato` | apply the patch printed by the script (section 2) |
| `sim: venv assente: ...` | run `~/vision_stack_deploy/setup_env.sh`, or set `VENV_ACTIVATE_SIM` |
| `sim: 'uav_roles.observer1' non leggibile` | the sim tree is not in `~/git/training`: set `SIM_TREE` |
| `Errore: directory FlightForge non trovata` | set `FLIGHTFORGE_DIR` |
| `Errore: stack grafico non idoneo` | NVIDIA driver not loaded. Fix the driver, or `ALLOW_INTEL_RENDERER=1` at your own risk (broken rendering, cameras that fail to start) |
| `Errore: interprete Python non trovato` | the sim tree venv is missing (section 2) |
| Observers run but publish no rays | a box must be confirmed by 3 detections in a row before a ray goes out. Look at `/uav1/debug_annotated/compressed` (observer overlay) and `/uav1/triangulation_debug_annotated/compressed` (rays) in `rqt_image_view`. If there are no boxes, it is the weights; if there are boxes but no rays, it is the TF (below) |
| Boxes but no rays / TF errors | `ros2 run tf2_tools view_frames` and check the frame in uav1's `CameraInfo` (trap 2) |
| Drones crash right after a command | wrong origin in `world_config.yaml` |

| `nvidia-smi` fails after an Ubuntu update | the new kernel was installed without its NVIDIA module. `sudo apt install linux-modules-nvidia-595-open-$(uname -r)` (use your driver version), then reboot |

## 7. Before a real flight

`ekf.kf_measurement_noise` in the deploy yaml is set to the **sim** value `0.0001`.
On the real drones it must be `0.0019`. The comment on that line says so.
