# Vision Stack Deploy

Two drones watch a third one. Each observer runs a YOLO detector on its own camera,
turns every detection into a **bearing ray frozen at image-capture time**, and publishes
that ray. A third process pairs the two rays, triangulates them, and runs the result
through an IMM filter to produce `/target/odometry_estimate`.

Bearing-only: no depth is used in the current configuration. Range comes entirely from
the geometry of two rays crossing, which is why almost everything in this document is
really about **pointing accuracy and clock sync**.

---

## If you have ten minutes

New here and flying soon? This is the whole path; every step links to its own section.

1. **[Install](#install)** — `bash setup_env.sh` on all three machines, then
   `git rev-parse --short HEAD` on each and confirm the three match. A mismatched `uav11`
   is the most common cause of a wasted flight.
2. **[Configure](#configure)** — open `config/runtime_stack.yaml` and check the handful of
   values in that section. What each number means, what happens if you move it, and which
   ones must be reviewed before a flight is in **[`docs/config_rationale.md`](docs/config_rationale.md)**
   — start at its *Check before every flight* table, which is the short version of the list
   below.
3. **[Preflight](#preflight)** — `python3 preflight_check.py`, then `--live` with both
   drones on the same flat ground. `NOT READY TO FLY` is a stop, not a warning.
4. **[Start order](#start-order)** — uav8 observer, uav11 observer, uav11 triangulation,
   ground monitor. In that order.
5. **[The ground gate](#the-ground-gate-before-you-record)** — `/tracker/fusion_mode` must
   reach `STEREO` on the ground with the target visible to both cameras. If it does not
   reach STEREO on the ground, it will not in the air.
6. **[After the flight](#after-the-flight)** — `bash run_prediction_evaluation.sh`.

**The five things that most often ruin a flight**, all covered in
[Configure](#configure) and [Preflight](#preflight):

| # | What | Why it matters |
|---|------|----------------|
| 1 | the three machines on different commits | silent, and it breaks everything downstream |
| 2 | `evaluation.shared_world_origin` vs. the real Z datum | a wrong datum is a constant bias on every ray, and nothing detects it at runtime |
| 3 | a stale `bag_profile` / `camera_tf` bias | the FCU's level reference moves at every boot, so last week's fit is 4–8° of pointing error |
| 4 | clocks more than ~50 ms apart | rays get paired that looked at the target at different moments |
| 5 | `uav_roles.target` naming the wrong drone | you will triangulate something, just not what you meant |

To watch what the two drones actually see, `vision.publish_debug_annotated_image` is
already `true`: the triangulation node draws its own ray in green and the other drone's in
red, on `/uav11/triangulation_debug_annotated/compressed` and
`/uav8/triangulation_debug_annotated/compressed`. Only the uav11 one actually renders —
read "the uav8 overlay is advertised but always empty" in Troubleshooting before you go
looking for the other.

---

## The one rule

**`config/runtime_stack.yaml` wins.** It is the source of truth for every runtime value. If
this README and the YAML disagree, the YAML is right and this README has a bug; fix it here.

The YAML holds the values and one short note each; **why** a value is what it is lives in
[`docs/config_rationale.md`](docs/config_rationale.md) — ranges, what moving each number does,
what every boolean switches on and off, and the measurements the numbers came from. That split
is deliberate: the file you open to find a number before a flight should not be an essay. Read
the rationale when you need to change something, not when you need to launch — and if you
change a value, change its explanation there too, since it is now the only copy.

For the field, `PREFLIGHT_CHECKLIST.txt` is the document you actually carry. It is in
Italian and it is more detailed than this file. Read [Known-wrong and known-inert](#known-wrong-and-known-inert)
before trusting it — it has three errors as of this writing.

---

## Who runs what

Three machines, four processes. Nothing is auto-started; you start each one by hand.

| Machine | Process | Command |
|---------|---------|---------|
| `uav8` (observer2) | detector → rays | `bash launch_observer_detection_node.sh` |
| `uav11` (observer1) | detector → rays | `bash launch_main_detection_node.sh` |
| `uav11` (also) | triangulation + IMM | `bash launch_triangulation_node.sh` |
| Ground station | live monitor | `bash launch_live_monitor.sh` |

`uav11` does double duty: it is an observer *and* the triangulating node. That is why it
gets two terminals.

Both observer launchers are three-line wrappers that set `OBSERVER_ROLE` and hand off to
`launch_observer_drone.sh`, which picks the node based on `launcher.drone_run_mode`:
`simple_detection` runs `nodes/test_yolov8_realtime_node.py`, anything starting with
`decentralized` runs `nodes/observer_node.py`.

The network is Zenoh (`rmw_zenoh_cpp`). It matters: `/tf_static` does **not** propagate
between drones over it. See [shared_world_origin](#datum-in-z-the-real-gate) below —
this single fact is why half the configuration looks the way it does.

---

## Install

Once per machine (both drones and the ground station):

```bash
git clone <this repo>
cd vision_stack_deploy
bash setup_env.sh          # creates .venv, installs requirements.txt
```

`requirements.txt` is small: `ultralytics`, `openvino`, `opencv-python-headless`,
`numpy<2`, `pyyaml`, `torch`, `torchvision`. ROS 2 Jazzy is expected at
`/opt/ros/jazzy/setup.bash`; every launcher sources it, falls back to any
`/opt/ros/*/setup.bash`, then sources `~/ros2_ws/install/local_setup.bash` and the local
`.venv` if they exist. Override with `ROS_SETUP`, `LOCAL_SETUP`, `VENV_ACTIVATE`.

**All three machines must be on the same commit.** `git rev-parse --short HEAD` on each.
This is the first item on the field checklist for a reason: a mismatched `uav11` is the
single most common cause of a flight that produces nothing usable.

Re-run `setup_env.sh` only when `requirements.txt` changed.

---

## Configure

Everything lives in `config/runtime_stack.yaml`. It is deliberately short — one line per
setting, one brief note saying what the value means or what range it takes. The reasoning
behind the numbers lives in [`docs/config_rationale.md`](docs/config_rationale.md); if a note
in the YAML is not enough, that is where the rest of it went.

These are the values to actually look at before a flight; the rest have defensible defaults.

```yaml
launcher:
  drone_run_mode: "decentralized_evaluation"   # simple_detection | decentralized | decentralized_evaluation
  use_sim_time: false                          # real drones publish no /clock — must be false

uav_roles:
  observer1: uav11      # main drone; also runs triangulation
  observer2: uav8
  target: uav2          # <-- CHECK THIS (see below)

vision:
  model_path: "dataset_tools/weights8/best_openvino_model"

bag_profile: "none"     # only valid value for a real flight
```

Four things to confirm by eye every single time:

1. **Which weights are you flying.** `dataset_tools/` holds `weights` through `weights8`
   plus `weights2_old` and `weights7_`. `preflight_check.py` verifies only that the path
   *exists*, never that it is the one you meant. Decide deliberately.

2. **`uav_roles.target`.** The file currently says `uav2`, which is the target in the
   Temesvar bags. The field checklist says `uav28`. One of them is wrong for your flight —
   resolve it on the ground, and remember the image topics name the drones in a *second*
   place in the YAML, so changing a role means changing both.

3. **`bag_profile: "none"`.** Any other value injects a `bearing_bias` fitted on one
   specific power-up of one specific airframe. The FCU's level reference is re-established
   at every boot, so a stale profile is 4–8° of pointing error — about 0.31 m per degree
   at 18 m.

4. **`shared_heading: false`.** `true` rewrites observer1's rays using observer2's yaw.
   That is valid only with both drones sitting still and parallel on the ground. Airborne
   it feeds any real heading difference straight into uav11's bearing error. The node
   prints a WARN at startup if you leave it on.

5. **The pixel knobs match the resolution you are flying.** Everything under
   `kalman_filter:` whose name ends in `_px`, plus `kf_sigma_a`, `kf_p_max` and
   `kf_bbox_area_ref`, are **absolute pixel quantities**. The active values are for
   1920×1080; each line's trailing comment carries the 1280×720 and 640×480 equivalents, so
   changing resolution is a copy from the comment. Fly 480 imagery on the 1080 column and
   every gate is three times too wide. (The August Temesvar bags are 640×480 — measured, not
   assumed.)

Currently off by deliberate choice, not by accident: `depth_fusion.enabled` (the USB-C
cable was swapped for USB 2.1 to stop RTK interference, and the link no longer carries an
aligned depth stream), `ekf.anisotropic_bearing_noise.enabled`, and `ekf.ml_refine`.

Everything else has a defensible default. What each one does, what range it lives in, and
what happens when you move it: [`docs/config_rationale.md`](docs/config_rationale.md).
Its *What the on/off switches actually do* section lists every boolean in the file with the
value it currently holds — worth two minutes before you flip one.

> The `ekf.*` and `kalman_filter.*` values come from a CMA-ES/CRLB pass and are **not yet
> validated on real hardware**. Treat the next flight as a filter validation, not a
> routine deploy.

---

## Preflight

Two stages, deliberately separate.

```bash
python3 preflight_check.py            # config only, no ROS needed — run this first
python3 preflight_check.py --live     # adds live attitude and topic checks
python3 preflight_check.py --live --seconds 10
```

The config stage catches settings that are correct for a bag replay and wrong for a real
flight. It prints `NOT READY TO FLY` and that is a stop. WARNs do not block but are worth
reading.

The live stage answers the one question no file can: how far off level each FCU booted.
Run it with both drones on the **same flat surface** and RTK fixed.

- roll or pitch beyond 1.0° → `<-- OFF LEVEL` (advisory)
- **differential** tilt obs2−obs1 beyond 1.0° → FAIL, blocking
- yaw disagreement beyond 2.0° → WARN

A differential tilt over 1.0° means you **reboot the worse FCU on level ground**. Do not
compensate in software: the bias is not repeatable between boots, which is exactly what
makes old `bag_profile` values useless.

### Clocks

Both observers stamp each measurement with image *capture* time, and pairing uses those
stamps with `triangulation.max_pair_age_sec: 0.20`. Two clocks 50 ms apart will pair rays
that looked at the target at different moments, and the geometry will be wrong with
nothing in the system noticing.

```bash
chronyc tracking      # Leap status must read Normal; offset under 10 ms is fine
chronyc sources -v    # chosen source needs '*', not '?' or '+'
```

Over ~50 ms is a stop.

### Datum in Z (the real gate)

Each drone has its own TF tree and its own `world_origin`. Historically they did not
coincide. The triangulation node used to bridge them through `utm_origin`, but that hop
needs `/tf_static` between drones, and **Zenoh does not carry it** — without the bridge,
every uav8 observation is discarded, no ray pair ever forms, and you sit on COAST
watching a single-drone solution.

`evaluation.shared_world_origin: true` skips the hop by declaring the origins identical.
Measure that before believing it. On **each drone via ssh** (not from the ground station —
`/tf_static` will not reach you there):

```bash
ros2 run tf2_ros tf2_echo uav11/utm_origin uav11/world_origin
ros2 run tf2_ros tf2_echo uav8/utm_origin uav8/world_origin
```

- Z difference under ~0.30 m → `shared_world_origin: true`
- over ~0.30 m → leave it `false` **and do not fly two drones**: with `false` uav8
  contributes nothing, and with `true` that gap becomes a silent constant bias on every
  ray.

Also confirm the frame name matches `evaluation.world_frame_suffix` (`world_origin`; some
configurations use `f9p_origin`):

```bash
ros2 topic echo /uav8/estimation_manager/odom_main --field header.frame_id --once
```

---

## Start order

The order matters: triangulation subscribes to both observers, and the first seconds fill
buffers and settle the datum.

1. **uav8** — `bash launch_observer_detection_node.sh`
   Wait for the periodic `Observer node: FPS=..., inference_avg=...ms`. Expect ~10 FPS on
   CPU. Check the logged YOLO model path is the one you chose.
2. **uav11** — `bash launch_main_detection_node.sh`
   Same checks. **The two FPS must be similar.** If uav8 runs at half of uav11, stereo
   pairs form rarely and everything downstream degrades.
3. **uav11, second terminal** — `bash launch_triangulation_node.sh`
   Wait for `Triangulation node initialization complete.` and the `Datum alignment ...`
   line.
4. **Ground station** — `bash launch_live_monitor.sh`

With `drone_run_mode: decentralized_evaluation`, step 3 also starts
`nodes/evaluate_prediction_node.py` alongside the triangulation node and waits for it to
finish writing on shutdown.

---

## The ground gate, before you record

With the target physically visible from **both** cameras:

```bash
ros2 topic echo /tracker/fusion_mode        # must reach STEREO
ros2 topic hz /target/odometry_estimate     # ~10 Hz (detection rate)
ros2 topic hz /target/predicted_trajectory  # ~10 Hz
ros2 topic hz /target/ground_truth_aligned
ros2 topic delay /uav11/target_observation  # and /uav8/...
```

If `fusion_mode` never reaches STEREO on the ground, it will not improve in the air. Stop.

If `/target/odometry_estimate` reads ~100 Hz while `fusion_mode` sits on COAST, you are
running code from before the coast-guard fix — wrong commit on uav11. Go back and align
the commits. It is the fastest way to detect that specific mistake.

If mean latency exceeds ~150 ms, raise `triangulation.coast_after_sec` to roughly 1.5×
the detection period and relaunch.

**Coast test:** cover one camera — `fusion_mode` should stay STEREO (or drop to
SINGLE_DEPTH only if the remaining drone has valid RGB-D depth and confidence > 0.60;
with `depth_fusion` off it goes straight to COAST). Cover both — COAST within ~0.15 s.
Uncover — it must return to STEREO.

Start the bag **before** takeoff, then confirm at t+10 s that it is growing:
`du -sh <bag_dir>`. A zero-byte bag at the end of the day is the classic way to waste one.

---

## After the flight

```bash
bash run_prediction_evaluation.sh
```

Artifacts land in `debug/evaluation/<timestamp>/` with `debug/evaluation/latest`
symlinked to the most recent run: `prediction_eval.json`, a per-sample CSV, a markdown
report, and two plots. `AUTO_OPEN_RESULTS=1` opens them. Record the STEREO /
SINGLE_DEPTH / COAST split and the RMSE while you still remember the flight.

---

## Offline replay (the development loop)

To work on the filter without flying:

```bash
./replay_bag_a.sh                      # play, monitor, score
./replay_bag_a.sh --no-images          # skip the two rqt windows
./replay_bag_a.sh --no-score           # don't record, don't score
./replay_bag_a.sh --start-offset 150   # skip further into the flight
```

It starts a Zenoh router if needed, runs the triangulation node against the bag under sim
time with `BAG_PROFILE=temesvar_A`, records the three result topics, and on Ctrl-C scores
the run into `<recording>/score.txt` via `dataset_tools/score_replay.py`. It runs on its
own `ROS_DOMAIN_ID=45` so it will not disturb a live session, and everything dies together.

You do not need the second "ground truth" bag — the triangulation node republishes
`/target/ground_truth_aligned` from the target's own odometry with the profile's Z datum
correction applied.

For the full evaluator report instead of just the score:
`DRONE_RUN_MODE=decentralized_evaluation ./replay_bag_a.sh` — it costs CPU and up to a
minute of shutdown while it renders plots, which is why it is not the default.

Two knobs worth turning by hand before reaching for anything heavier:

```bash
KF_PROCESS_NOISE_VEL=3.0 MAX_COAST_SEC=0.5 ./replay_bag_a.sh --no-images
```

Beyond two or three manual attempts, tuning belongs in a CMA-ES run, not in a terminal.

### Self-checks

```bash
python3 -m py_compile nodes/*.py
python3 nodes/test_coast_singer.py            # closed-form Singer coast propagation
python3 nodes/test_live_monitor_gt_fallback.py
python3 nodes/test_debug_overlay_sides.py     # observer2's overlay reads the right ray
```

---

## Environment overrides

Every launcher reads its values from the YAML but lets the environment win, which is what
makes A/B runs cheap.

`launch_triangulation_node.sh`: `OBSERVER1_NAME`, `OBSERVER2_NAME`, `TARGET_NAME`,
`USE_SIM_TIME`, `DRONE_RUN_MODE`, `BAG_PROFILE`, `KF_PROCESS_NOISE_ACC`,
`KF_PROCESS_NOISE_VEL`, `KF_MEASUREMENT_NOISE`, `KF_INITIAL_COVARIANCE`,
`MAX_PAIR_AGE_SEC`, `OBSERVATION_BUFFER_SEC`, `TRIANGULATION_MIN_BASELINE_M`,
`MAX_TRIANGULATION_ERROR_M`, `PROCESSING_RATE_HZ`, `MAX_COAST_SEC`.

`launch_observer_drone.sh`: `UAV_NAME`, `OBSERVER_ROLE`, `MODEL_PATH`, `USE_SIM_TIME`.

`replay_bag_a.sh`: `BAG`, `REC_DIR`, `DRONE_RUN_MODE`, `RMW_IMPLEMENTATION`,
`ROS_DOMAIN_ID`.

Extra `--ros-args` passed to any launcher are forwarded to the node.

---

## Repository layout

| Path | Purpose |
|------|---------|
| `config/runtime_stack.yaml` | **Source of truth** for all runtime values |
| `preflight_check.py` | Two-stage pre-flight gate (`--live` for attitude/topics) |
| `PREFLIGHT_CHECKLIST.txt` | The field checklist (Italian); print it |
| `nodes/observer_node.py` | Per-drone detection, ROI tracking, frozen bearing rays |
| `nodes/triangulation_node.py` | Ray pairing, WLS triangulation, IMM filter, GT republication |
| `nodes/evaluate_prediction_node.py` | Post-flight evaluator → `debug/evaluation/` |
| `nodes/live_monitor.py` | Ground-station plots |
| `nodes/imm_image_overlay_node.py` | Debug overlay of the 3D estimate on the image |
| `nodes/test_yolov8_realtime_node.py` | Local detection path for `simple_detection` mode |
| `nodes/test_coast_singer.py` | Self-check: Singer coast propagation |
| `nodes/test_live_monitor_gt_fallback.py` | Self-check: monitor's GT fallback |
| `nodes/test_debug_overlay_sides.py` | Self-check: observer2's debug overlay picks the right ray |
| `docs/config_rationale.md` | Why every value is what it is; read before changing one |
| `launch_main_detection_node.sh` | Observer on uav11 (`observer1`) |
| `launch_observer_detection_node.sh` | Observer on uav8 (`observer2`) |
| `launch_observer_drone.sh` | Shared implementation of the two above |
| `launch_triangulation_node.sh` | Triangulation + IMM (+ evaluator in `*_evaluation` mode) |
| `launch_live_monitor.sh` | Ground-station monitor |
| `run_prediction_evaluation.sh` | Standalone evaluator run |
| `replay_bag_a.sh` | One-command offline replay and scoring |
| `dataset_tools/score_replay.py` | Scores a recorded replay |
| `dataset_tools/weights*/` | Detector weights (`model_path` selects one) |
| `setup_env.sh`, `requirements.txt` | Python environment |

---

## Known-wrong and known-inert

### Errors in `PREFLIGHT_CHECKLIST.txt`

The checklist is otherwise the best document here, but three items are wrong:

- **`ekf.normalize_crlb` is NOT a no-op.** The checklist says twice that it "isn't read by
  any line of code" and tells you to ignore it. It is read at `triangulation_node.py:1904`,
  declared at `:2075`, stored at `:2133`, passed to `triangulate_wls()` at `:3748`, and
  branched on at `:938`. `preflight_check.py:102` carries a comment noting the branch was
  deleted once and restored, "so the value matters." It decides whether `R` keeps its raw
  CRLB scale (`false`, current) or is rescaled to `trace = 3*kf_measurement_noise` (`true`).
- **`nodes/test_coast_guard.py` and `nodes/test_anisotropic_bearing_noise.py` do not
  exist.** The real self-tests are `nodes/test_coast_singer.py` and
  `nodes/test_live_monitor_gt_fallback.py`.
- **Target name.** The checklist says `target = uav28`; the YAML says `uav2`.

### Config keys that do nothing

- **`ekf.kf_adaptive_cov_max_multiplier`** (50.0) — this one **is** in the YAML, which makes
  it the dangerous kind. Read at `triangulation_node.py:1922`, declared at `:2086`, stored
  at `:2149`, and then never used by anything. It is the saturation cap of a NIS-driven
  adaptive covariance scheme that was never written: the filter does compute the Normalised
  Innovation Squared, and both call sites throw it away. Turning this knob has no effect
  whatsoever. Full analysis in `docs/config_rationale.md`.
- **`ekf.imm_transition_prob`** — read at `triangulation_node.py:1896`, declared at `:2069`
  and stored at `:2125`, then never consumed; the source marks it `# legacy, unused`. The IMM
  Markov matrix is built from `ekf.imm_p_stay_cv` (0.75) and `ekf.imm_p_return_to_cv` (0.30).
  Change those. Not present in this repository's YAML.
- **`vision.roi_crop_imgsz`** — inert. `observer_node.py:1024` uses it only as a fallback
  when `roi_reference_imgsz` is absent, and it is not absent (`vision.roi_reference_imgsz: 256`).
  The parameter is declared at `:1150` with the same default, so the override branch at
  `:1232-1237` never fires. Not present in the current YAML.
- **`launcher.auto_start_node`** and **`evaluation.auto_start_with_launch_sim`** — present in
  this YAML **for parity only**: nothing in this repository reads either one, and both are
  labelled as such in the file. **They are not dead everywhere, so do not go delete them from
  the other tree.** The simulation and bag-replay setup at
  `~/git/training/mrs_uav_flightforge_simulator/tmux/one_drone` keeps its *own separate copy*
  of `runtime_stack.yaml`, where both keys are live: `auto_start_node` chooses which node the
  generated tmux session starts — `none`, `detection`, `triangulation` or `decentralized`
  (`generate_runtime_stack.py:397`, overridable with `AUTO_START_NODE_OVERRIDE`) — and
  `auto_start_with_launch_sim` is read by `launch_sim.sh:149` to decide whether the prediction
  evaluator comes up alongside the live monitor. `./launch_sim.sh` exists there and is still
  in use; it simply is not part of this deploy repo. Keeping the keys here means the two files
  stay diffable, which is the only reason they are not deleted.

### Known broken, do not chase in the field

**Asymmetric re-acquisition after target loss.** When the visual track is lost, the
observer falls back to cropping a narrow ROI around the current 3D estimate instead of
re-running full-frame detection. That fallback works **only on uav11**. On uav8 the
required transform crosses two disjoint TF trees, always raises, and the exception is
swallowed silently — no log, no counter. Expected in flight: after an occlusion, uav8
re-acquires systematically slower than uav11. It is a known, analysed degradation with the
fix deliberately deferred until after the next flight, because completing it needs the Z
datum measurement from the preflight. If you see uav8 reacquiring slowly, this is it —
stop looking.

The debug overlay used to exist for uav11 only, which is why this presented as a one-drone
problem. Since 2026-09-12 a publisher exists for both, but the uav8 one never emits a
frame — see "the uav8 overlay is advertised but always empty" in Troubleshooting. Watch
`/uav8/detection2/compressed` and the node's `DBGIMG` counters instead.

---

## Troubleshooting

| Symptom | Cause |
|---------|-------|
| `fusion_mode` stuck on COAST while a correct box is visible on `/uav8/detection2` | Two distinct causes. **(a)** uav8's rays never arrive — check `shared_world_origin` and the Z datum first. **(b)** The rays are never *published*: `observer_node.py:2774` publishes a ray only once the image-space track is CONFIRMED, and confirmation needs 3 **literally consecutive** detections (`kalman_filter.kf_n_confirm: 3`) — a single frame without a detection resets the counter at `:647`. With intermittent detections the track stays TENTATIVE forever, so you see a correct box while triangulation logs `MONO: no uav8 observation`. A visible box is not yet a ray. |
| Model not found | `vision.model_path` is resolved from the repo root; confirm `dataset_tools/weights8/best_openvino_model` exists |
| Live monitor shows nothing | `evaluation.odom_topic_suffix` must match the platform. `estimation_manager/odom_main` is **correct** for these drones; `hw_api/ground_truth` is simulation-only |
| Nodes start but never process | `launcher.use_sim_time: true` on real drones — they wait forever for a `/clock` nobody publishes |
| Estimate jumps between two trails | A node from a previous run is still alive and publishing on `/target/odometry_estimate` |
| TF lookup fails | Check `vision.publish_camera_static_tf`. It must stay `true` on the drones: the RealSense driver publishes its chain un-namespaced with no link to `fcu`, so with it `false` the chain does not exist and the observer produces no rays |
| No ROS found | Set `ROS_SETUP=/opt/ros/<distro>/setup.bash` |
| the uav8 overlay is advertised but always empty | Not a regression, and not reachable from config. Drawing uav8's ray needs `uav8/rgbd_color_optical_frame`, which the node — running on uav11 only — never receives: Zenoh does not propagate `/tf_static` between drones, and bag A carries it for `uav11/*` only. Those frames are dropped at `triangulation_node.py:2632` and counted in the node's `DBGIMG ... skip: no_frame=N` line. Use `/uav8/detection2/compressed` instead; detail in [`docs/config_rationale.md`](docs/config_rationale.md) |
| `replay_bag_a.sh` opens no rqt window, log shows an `ImportError` on PyQt5 | The script activates `.venv`, which is built with `include-system-site-packages = false` and has no PyQt5, while rqt is a system package. Fixed 2026-09-14 by starting rqt with the pre-venv `PATH`; if you meet it elsewhere, start the Qt tool outside the venv |
