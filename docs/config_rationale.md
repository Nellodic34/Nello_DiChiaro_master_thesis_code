# Reading `runtime_stack.yaml`

This file explains the configuration so the YAML itself can stay short. It is organised
worst-consequence-first:

1. **[Check before every flight](#check-before-every-flight)** — get these wrong and the
   stack either produces nothing or produces confident nonsense.
2. **[What the on/off switches actually do](#what-the-onoff-switches-actually-do)** — plain
   description of the booleans you will meet.
3. **[Knobs that are not knobs](#knobs-that-are-not-knobs)** — settings that look live and
   are not.
4. **[Every setting, in file order](#every-setting-in-file-order)** — every value in the
   YAML, each one sitting next to the prose that justifies it: ranges, what moving it does,
   and the measurements the number came from.

This repository's `runtime_stack.yaml` is an **independent copy**, not shared with the
simulation/bag tree at `~/git/training/mrs_uav_flightforge_simulator/tmux/one_drone`. The
two have diverged; editing one does not affect the other.

---

## Check before every flight

These are the settings whose wrong value is either invisible or catastrophic. Everything
else can be tuned later; this list is the pre-flight gate.

| Setting | Must be | Why it bites |
|---|---|---|
| `evaluation.shared_world_origin` | `true` **only if** every drone was configured with the same world origin, **including z** | This is the single most common cause of "uav8's rays never arrive". Zenoh does not propagate `/tf_static` between drones, so nothing detects a mismatched datum at runtime — a wrong z becomes a silent constant bias on every ray. Verify physically: park the drones on the same flat ground and compare `z` from `estimation_manager/odom_main`; `preflight_check.py --live` prints exactly this. |
| `uav_roles.shared_heading` | **`false` to fly** | Static-bench feature only. It rotates observer1's rays using observer2's yaw, which is valid only while both airframes sit parallel on the ground. Airborne, any real heading difference goes straight into observer1's bearing error. The node prints a warning at startup if it is left on. |
| `uav_roles.observer1` / `observer2` / `target` | the actual drone names | Everything downstream — topic names, TF frames, role selection in the launchers — is derived from these. |
| `launcher.use_sim_time` | `false` on real drones | Real drones publish no `/clock`. Set it true only for sim or bag replay, where a clock source exists. |
| `vision.main_image_topic` / `observer_image_topic` | the real camera topics | `main_*` is observer1, `observer_*` is observer2. A typo here yields no detections at all from that drone. |
| `vision.publish_camera_static_tf` | `true` on drones | The RealSense driver publishes its own chain un-namespaced and with no link to `fcu`. With this false the TF chain from `fcu` to the camera does not exist, so the observer produces **no rays whatsoever**. |
| `camera_tf.*` bearing bias / `bag_profiles.bag_profile` | empty / `"none"` for a real flight | These are per-boot, per-flight survey constants fitted against one specific take-off. Carrying a profile forward injects several degrees of pointing error — and at 18 m, one degree is about 0.31 m. |
| resolution-coupled pixel knobs (`kalman_filter` section) | matched to the stream you actually fly | These are absolute pixel quantities and only mean anything at the resolution they were tuned at. The active column is for 1920×1080; the trailing comments carry the 1280×720 and 640×480 equivalents. Flying a different resolution without rescaling leaves the gates several times too wide or too narrow. |
| `vision.model_path` | weights that exist on the drone | Relative paths resolve from the repo root. |
| `ekf.kf_mahalanobis_threshold` | understand it before touching | Currently `34.0` — chi²(3, 0.999), far above the code default of 9.0. It was raised to stop the filter rejecting valid braking frames, which is a symptom of an over-tight `R`. See [Knobs that are not knobs](#knobs-that-are-not-knobs). |
| `depth_fusion.enabled` | `false` with the current cabling | The USB 2.1 cable has no bandwidth for an aligned depth stream. Leaving it true makes `preflight_check.py` fail and buys nothing. |

---

## What the on/off switches actually do

Every boolean in the file, with the value it currently holds. A master switch set `false`
means the code path is not merely idle but absent — no subscriber, no publisher, no CPU
cost — which is why turning one on is never free and turning one off never "leaves it
running quietly".

### Diagnostics — safe to flip, the cost is bandwidth and CPU

- **`vision.publish_annotated_image`** *(now `true`)* — `true`: the observer publishes its
  camera image with the detection boxes drawn on it. `false`: that topic does not exist.
- **`vision.publish_debug_annotated_image`** *(now `true`)* — `true`: the triangulation node
  publishes, **for each observer**, that drone's camera image with the bearing rays drawn on
  it — its own ray in green, the other drone's ray in red, plus a one-line HUD reading
  `STEREO <peer> dt=… …`, or `MONO: no <peer> observation` when the other drone contributed
  nothing at that instant. `false`: no overlay topics, no image subscriptions, no image
  decoding at all. This is the single most useful thing to look at when one drone seems not
  to be contributing.
- **`vision.publish_detector_evaluation_packets`** *(now `true`)* — `true`: the observer emits
  extra telemetry for the evaluator. `false`: the evaluator has no observer-side data and the
  observer does no evaluation work at all.
- **`vision.publish_rviz_markers`** *(now `false`)* — `true`: predicted trajectory and
  position markers are published for RViz. `false`: no markers.
- **`triangulation.publish_evaluation_packets`** *(now `true`)* — `true`: publish
  `/target/triangulation_evaluation`, which the live monitor and the offline scoring read.
  `false`: those tools go blind; the estimate itself is unchanged.
- **`triangulation.pairing_log_best_cost`** *(now `false`)* — `true`: log the winning pairing
  cost and its three terms at a throttled rate. `false`: silent. Pure debug, no behaviour
  change either way.

### Structural — get one of these wrong and you get no output at all

- **`vision.publish_camera_static_tf`** *(now `true`, must stay so on drones)* — `true`: this
  node publishes the `fcu`→`rgbd_link` transform. `false`: the TF chain from the flight
  controller to the camera does not exist, so the observer produces **no rays whatsoever**.
- **`launcher.use_sim_time`** *(now `false`)* — `true`: follow `/clock`. `false`: use wall
  time. Real drones publish no `/clock`, so `true` on a real drone stalls every node.
- **`evaluation.shared_world_origin`** *(now `true`)* — `true`: assume every drone shares one
  world origin, so positions from different drones can be compared directly. `false`: they
  cannot be, and cross-drone triangulation is not possible. See the pre-flight table: this is
  an assumption about the *physical setup*, not a preference.
- **`uav_roles.shared_heading`** *(now `false`, must stay so to fly)* — `true`: observer1's
  rays are rotated using observer2's yaw. `false`: each drone uses its own. See the pre-flight
  table.
- **`vision.allow_tf_fallback`** *(now `true`)* — `true`: when the exact frame-time transform
  is unavailable, the observer still publishes a ray using the latest transform it has.
  `false`: that detection is dropped. `true` trades a little pointing accuracy for far fewer
  gaps; `max_tf_delta_ms` is what keeps the fallback honest.
- **`kalman_filter.kf_enabled`** *(now `true`)* — `true`: each observer runs its image-space
  Kalman tracker, which is what rejects false positives and requires `kf_n_confirm`
  consecutive detections before a ray is published. `false`: every raw detection becomes a
  ray, including the bad ones.

### Estimator behaviour — flip only with a reason

- **`ekf.anisotropic_bearing_noise.enabled`** *(now `false`)* — `false`: the bearing
  uncertainty is a *circular* cone of width `kf_measurement_noise`, identical in azimuth and
  elevation. `true`: the cone becomes an *ellipse* set by `sigma_az_deg` and `sigma_el_deg`,
  and `kf_measurement_noise` is then ignored for the visual term. The two axes genuinely have
  different error sources on this airframe, which is what the ellipse is for — but check the
  ellipse is the right way round before trusting it.
- **`ekf.ml_refine`** *(now `false`)* — `false`: keep the closed-form weighted-least-squares
  answer. `true`: run a few Gauss–Newton steps on the true angular residual afterwards, which
  removes the bias inherent to the closed form. It starts from the WLS answer and falls back
  to it if a step diverges, so it cannot do worse; it costs a few small solves per frame.
- **`ekf.normalize_crlb`** *(now `false`)* — `false`: the covariance handed to the filter is
  the raw geometric bound, so the filter's trust in a fix follows the actual geometry — wide
  at long range, tight when the rays cross near a right angle. `true`: same shape, rescaled to
  a fixed overall size (legacy behaviour). **This is not a no-op**, whatever the field
  checklist says.
- **`triangulation.pairing_prediction_enabled`** *(now `true`)* — `true`: pick which two rays
  to pair using time, geometry **and** agreement with the current prediction. `false`: pair
  purely by closest timestamp.
- **`vision.roi_crop_enabled`** *(now `true`)* — `true`: when a valid predicted centre exists,
  run the detector on a crop around it — faster and better at range. `false`: always run on
  the full frame.
- **`tight_coupling.enabled`** *(now `false`)* — `true`: the 3D estimate projects back onto
  each observer's image plane as an extra gate, so detections that pass the local 2D gate but
  fall outside the 3D ellipse are rejected. `false`: pure loose coupling, each observer
  decides alone. `true` is sharper against look-alike targets and correspondingly better at
  locking onto its own mistake, so it disables itself when the 3D estimate is stale or
  uncertain.
- **`kalman_filter.kf_ego_motion_enabled`** *(now `true`)* — `true`: the image-space tracker
  compensates for the observer's own motion between frames. `false`: it does not, and the
  track lags whenever the observer manoeuvres.
- **`kalman_filter.kf_cam_aware_gating_enabled`** *(now `true`)* — `true`: the 2D gate scales
  with the target's apparent size and distance. `false`: one fixed gate at every range.
- **`observer_odom_uncertainty.use_odom_uncertainty`** *(now `false`)* — `true`: fold each
  observer's own position uncertainty into its ray's covariance. `false`: ignore it. **Must
  match whatever the tuning ran with**, or the covariance scale differs between tuning and
  flight. Bags carry no odometry covariance, so tuning on them with `true` produces the wrong
  scale.
- **`observer_odom_uncertainty.distance_dependent_noise`** *(now `true`)* — `true`: the ray's
  width grows with distance (a cone). `false`: constant width (a cylinder).
- **`observer_self_rejection.enabled`** *(now `true`)* — `true`: each observer discards
  detections that land on the *other observer* rather than on the target, using the sibling's
  odometry to project its body onto the image. `false`: no subscriber, no overhead — and
  nothing stops the detector from locking onto the other drone.
- **`observer_self_rejection.rejection_only_if_closer`** *(now `false`)* — `false`:
  unconditional geometric exclusion — anything projecting onto the sibling's body is rejected,
  regardless of range. `true`: reject only when the sibling is actually nearer than the
  estimated target. `true` **fails open** whenever the 3D estimate is missing or stale, so
  `false` is the right choice for two drones flying near each other.

### Depth — the whole branch is currently off

- **`depth_fusion.enabled`** *(now `false`)* — `true`: subscribe to the aligned depth stream
  and use range as an extra 1-D measurement. `false`: no depth subscriber, no depth code path,
  pure bearing-only. It is off because the USB 2.1 cable (fitted to stop RTK interference)
  cannot carry an aligned depth stream; leaving it `true` makes `preflight_check.py` fail.
  **The cost of `false`:** the single-observer fallback needs a valid depth sample, so when one
  observer is occluded the filter now only coasts instead of updating on one ray.
- **`depth_fusion.gating_enabled`** *(now `false`)* and
  **`depth_fusion.fusion_use_range_constraint`** *(now `true`)* — both are inert while
  `depth_fusion.enabled` is `false`. They describe, respectively, the occluder-rejection gate
  and whether an accepted depth sample is applied as a range update.

---

## Knobs that are not knobs

Three settings in this stack looked like tuning parameters and were not: one is **still
inert**, one has since been **wired up**, and one has been **removed**. They are documented
here rather than in the YAML because the explanation is longer than any comment should be,
and because a knob that does nothing is a safety problem: whoever turns it will attribute
the result to the value.

### `ekf.kf_adaptive_cov_max_multiplier: 50.0` — designed, instrumented, never connected

**Status: live in the YAML, read, stored, and used by nothing.**

The read chain is complete and looks healthy — default at `triangulation_node.py:1795`,
YAML override at `:1922`, `declare_parameter` at `:2086`, stored as
`self.kf_adaptive_cov_max_multiplier` at `:2149` — and then the attribute is never
referenced again.

The other half of the missing feature is just as visible. The IMM update computes the
Normalised Innovation Squared at `:1600` and returns it as the fifth value at `:1629` and
`:1663`. Both call sites discard it:

```python
x_comb, self.combined_cov, self.imm_probs, accepted, _ = self.imm.predict_and_correct(...)
```

at `:3610` and `:3783`. So the measurement and the cap of an innovation-based adaptive
covariance scheme both exist, and the rule that would connect them was never written.

### What it was meant to be

This is textbook innovation-based adaptive estimation (covariance matching): scale the
measurement covariance so the filter's assumed innovation covariance matches the innovations
actually observed. For an `n_z`-dimensional measurement, a consistent filter satisfies

```
NIS = nu^T S^-1 nu  ~  chi2(n_z),    E[NIS] = n_z   (here n_z = 3)
```

and the canonical rule is

```
s_k = NIS_k / n_z
R_k <- min(s_k, c_max) * R_k        when s_k > 1
```

`kf_adaptive_cov_max_multiplier` is precisely `c_max`, the saturation cap that stops a
single bad frame from blowing R up without bound. The name and the value only mean anything
inside that scheme, which is good evidence this is what was intended.

### Why this stack genuinely needs it

`R` reaches the filter from `triangulate_wls` as the Cramér–Rao bound, `inv(Lambda)`. A CRLB
is a *lower* bound — the best any unbiased estimator could achieve, not what this estimator
achieves. With `normalize_crlb: false` the raw bound is handed to the IMM, so `R` is
systematically optimistic.

The symptom is already written into the config. `kf_mahalanobis_threshold` sits at **34.0**
with the comment *"RAISED massively. With normalize_crlb=false, R is tight. If this is too
small, the filter will reject valid braking frames and coast 10m!"* — where the node's own
default is 9.0. Those are chi2(3, 0.999) and roughly chi2(3, 0.97). Widening the gate to
0.999 treats the symptom: it stops the filter rejecting good frames, but `R` is still too
small, so the Kalman gain still under-weights every measurement and the estimate still lags.
Scaling `R` to its measured consistency level treats the same disease at the cause, and
would let the gate return to a defensible value instead of one chosen to paper over a
mis-scaled covariance.

### Why it should not be wired blind

- It changes the closed-loop behaviour of a filter that flies. This is not comment cleanup.
- **It cannot be validated in simulation here.** The simulated bearing noise is far more
  optimistic than reality, which drives NIS down by orders of magnitude; any NIS-driven rule
  is inert or mis-calibrated against sim data. Validation has to happen on recorded flights.
- `Q` is independently suspect: the dominant error in this stack is extrapolation during
  coast, not geometry. Inflating `R` while `Q` is too small pushes the filter *further*
  toward trusting its own prediction — the exact failure already observed. `R` and `Q` have
  to be judged together, on the same data.

### Recommended order

1. **Stop discarding the NIS.** Name the fifth return value and log it at `:3610` and
   `:3783`. Two lines, zero behaviour change, and it is the prerequisite for every step
   below.
2. Fly or replay, and collect the NIS distribution. A consistent filter gives mean 3.
3. If the measured mean is much larger than 3 — expected, since `R` is a lower bound —
   prefer a **static** correction first: a constant scale mismatch is best fixed once, by
   `kf_measurement_noise` or by the `normalize_crlb` choice, not by an adaptive loop. Only
   if the mismatch varies with geometry or range does the adaptive rule earn its complexity,
   and then `c_max` is this knob and the feature ships default-off.
4. Only after step 3 does `50.0` mean anything.

Until step 1 exists, this is documentation rather than configuration. It has been left in
the YAML on purpose, pending that decision.

**References.** Mehra, "On the identification of variances and adaptive Kalman filtering",
IEEE TAC 15(2), 1970; Mehra, IEEE TAC 17(5), 1972; Maybeck, *Stochastic Models, Estimation
and Control* Vol. 2, 1982, Ch. 10; Bar-Shalom, Li & Kirubarajan, *Estimation with
Applications to Tracking and Navigation*, 2001, Sec. 5.4 (filter consistency, NIS/NEES).

### `vision.observer_image_topic` → `observer2_image_topic` — was inert, wired up 2026-09-12

The natural question is: *isn't the image/camera_info subscription already determined by
`observer_image_topic` / `main_image_topic`?* Yes — but there are **two different consumers**
of those keys, and only one of them is symmetric.

**1. The observer side — production path, symmetric, works.** Each observer drone runs
`observer_node.py` and subscribes to its own camera. The role decides which key it reads:
`observer1` → `vision.main_image_topic`, `observer2` → `vision.observer_image_topic`, and
identically for camera_info at `observer_node.py:1005-1006`. Both roles are fully handled.
Nothing is missing here, and this is the path that produces bearing rays.

**2. The triangulation side — debug overlay, formerly observer1 only.**
`triangulation_node.py` *also* subscribes to raw images, for exactly one purpose: drawing
the reprojected rays onto the picture. The configuration had always been read for both
roles, but only observer1 got a subscription, a publisher and a world frame; three explicit
guards (`if source_name != self.observer1_name: return`) dropped everything else, so
`self.observer2_image_topic` was assigned and never read again. Changing
`vision.observer_image_topic` correctly changed what uav8's *observer* subscribed to, and
had no effect whatsoever on the overlay.

**This was fixed on 2026-09-12.** Both observers now get image and camera_info
subscriptions, their own world frame, and their own publisher at
`/{drone}/triangulation_debug_annotated/compressed`. There is a new optional key,
`vision.observer_camera_info_topic`, which overrides observer2's CameraInfo topic; left
empty (the default) it is derived from `observer_image_topic` exactly as observer1's is
from `main_image_topic`.

**The subtlety worth knowing before touching that code.** A stored `MatchedObservationPair`
always holds observer1 on its `local` side and observer2 on its `remote` side. Drawing
observer2's overlay therefore has to match pairs on the *remote* stamp and report the
*opposite* end of the pair, or the red "other drone" ray belongs to a different instant
than the green one — a wrong picture that looks entirely plausible. That is what
`_find_closest_matched_pair(..., match_remote_side=True)` is for, and what
`nodes/test_debug_overlay_sides.py` checks.

**What was never affected: triangulation itself.** The measurement path was always
symmetric — `observer1_measurement_sub` and `observer2_measurement_sub` — so uav8's rays
always arrived and were fused normally. The inert knob cost diagnostics, not accuracy.

**Why it mattered.** This is why uav8's slow re-acquisition used to be hard to see: when
uav8's image-space track sits in TENTATIVE it publishes no ray, and with no uav8 overlay a
two-drone problem presented as a one-drone problem.

**What the fix did not buy — measured 2026-09-14.** The publisher is symmetric now, but
drawing a ray still needs a world→camera transform for the drone being drawn, and that
comes from TF (`_resolve_camera_frame`, `triangulation_node.py:2576`). The triangulation
node runs on uav11 only (`PREFLIGHT_CHECKLIST.txt`, step 7.3) and
`uav8/rgbd_color_optical_frame` never reaches it: Zenoh does not propagate `/tf_static`
between drones — the same fact `evaluation.shared_world_origin` exists to work around —
and bag A carries `/tf_static` for `uav11/*` only. Every uav8 frame is therefore dropped
at `:2632` and counted in `DBGIMG ... skip: no_frame=N`, so the topic is advertised and
stays empty, in replay and in flight alike. Closing this means composing the observer's
own odometry with `camera_tf.observer2` instead of looking TF up — note `camera_tf` is
read by `observer_node.py` and never by the triangulation node. Deferred deliberately: it
is a diagnostic, and it does not touch the measurement path.

### `observer_odom_uncertainty.min_bearing_distance` — removed 2026-09-12

Formerly `runtime_stack.yaml:552`, with the read chain at `triangulation_node.py:1832`,
`:2007-2009`, `:2065` and `:2174`. It was stored as `self.min_bearing_distance` and **never
passed to `triangulate_wls`** — the call site does not include it, so the function's own
default always applied. Because the YAML value and that default were both `0.5`, behaviour
never differed; but editing the YAML value would have done nothing, which is the whole
problem.

Removed from the YAML and from the node. The `min_bearing_distance` argument of
`triangulate_wls` itself is **kept**: it is a live clamp, used at `:1202` and `:1209`.

---

## What the filter does to the measurements — measured 2026-09-14

One replay of bag A (`DRONE_RUN_MODE=decentralized_evaluation ./replay_bag_a.sh`, 176.6 s
of ground truth), scoring `/target/triangulation_raw` and `/target/odometry_estimate`
against `/target/ground_truth_aligned` — the datum-corrected truth the scorer itself uses —
and splitting the horizontal error along and across the target's direction of travel:

| | n | RMSE 3D | median | p90 | max | along-track @ >3 m/s | cross-track @ >3 m/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `triangulation_raw` | 563 | 1.333 m | 0.464 m | 1.990 m | 8.315 m | −0.53 m | −0.03 m |
| `odometry_estimate` | 1003 | 2.228 m | 0.625 m | 3.925 m | 9.712 m | −2.59 m | +0.06 m |

The filter adds almost no lateral error: median absolute cross-track is 0.13 m raw against
0.18 m filtered. The entire penalty is along-track, it has one sign — the estimate sits
*behind* the target — and it is a function of target speed: +0.06 m below 0.5 m/s, −0.64 m
at 1–2 m/s, −2.59 m above 3 m/s. Expressed as a delay that is 0.14 s for the raw
measurements and 0.71 s for the filter. The excursion near t+40 s, where the RMSE gap comes
from, is the stretch where the target is fastest; the filter is not noisier than the
geometry, it is late.

Two independent statistics agree. The run scorer reports an effective gain of 0.22 over 563
paired measurements arriving at ~3.2 Hz, which predicts a time constant of
1/(0.22 × 3.2) ≈ 1.4 s, the same order as the measured lag.

This is **not** the coast path, which is the intuition worth discarding: RMSE while coasting
is 1.696 m against 3.074 m while actively fusing, and ANEES is 3.36 against an expected 3.0,
so the reported covariance is roughly honest on average. The estimator is under-weighting
its own measurements — measurements whose median error, 0.464 m, is better than the filtered
output's 0.625 m. Suspects, in order: `kf_process_noise_vel: 2.0` (CV acceleration std-dev)
against a target that genuinely manoeuvres above 3 m/s; `imm_p_stay_cv: 0.75` holding the
mixture on the constant-velocity model; and `normalize_crlb: false`, which leaves R at a raw
CRLB scale that over-states how bad the measurements actually are.

**Which of the three it is — measured, not argued, 2026-09-14.** Same bag, same everything,
only `KF_PROCESS_NOISE_VEL=8.0` passed to the launcher instead of the configured 2.0:

| | along-track @ >3 m/s | lag | RMSE 3D | median | p90 | max | effective gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `kf_process_noise_vel: 2.0` | −2.59 m | 0.71 s | 2.228 m | 0.625 m | 3.925 m | 9.712 m | 0.22 |
| `kf_process_noise_vel: 8.0` | −1.35 m | 0.37 s | 2.000 m | 0.864 m | 2.626 m | 8.409 m | 0.37 |

Quadrupling the CV acceleration noise roughly halves the lag and cuts the tail, and costs
0.24 m of median accuracy — a bias/variance trade. So the mechanism is the **process
model**, not the measurement model: the filter's Q is too small for a target that genuinely
manoeuvres, and R is at worst a second-order contributor.

Control: the raw measurements are statistically identical across the two runs (RMSE 1.333
vs 1.355 m, median 0.464 vs 0.462 m, 563 measurements in each), so nothing but the filter
moved. Both cover the same scoring window. Worth knowing before you compare your own runs:
the bag is 227 s long, the default `START_OFFSET=105` leaves ~102 s of it after the
recorder starts, and that is the entire stretch in which an estimate exists — the first ray
pair lands at +107 s. Starting earlier (`--start-offset 30`) records more *ground truth*,
not more estimate, and the extra GT is what makes two runs look like different lengths.

**Where the too-small Q comes from — and it was already written down.** The `ekf` notes
below carry this warning:

> WARNING: `kf_process_noise_*` and `kf_mahalanobis_threshold` were CMA-ES-tuned against
> the old `0.0002`, so the gate is looser than it was tuned to be.

`kf_measurement_noise` is now `0.0019`, sized from the *full* attitude budget — the camera
is bolted to the FCU with identity rotation, so attitude error enters the bearing 1:1 and
dominates it at 1.5–3.9° measured — instead of from the detector alone. That is a 9.5×
increase in assumed angular **variance**, σ from 0.81° to 2.5°. Q was never re-optimised
against it. R rose almost an order of magnitude, Q stayed where CMA-ES had put it, and the
gain — P/(P+R) in spirit — fell with the ratio. An effective gain of 0.22 is that
arithmetic, not a bug.

This also predicts the repair. Restoring the old Q/R ratio needs the process-noise
*variance* up 9.5×, so the std-dev up √9.5 ≈ 3.1×: `kf_process_noise_vel` ≈ 6.2 m/s²
against the configured 2.0. The measured best region is ~8, and this document's own
prescription for re-running CMA-ES already asks for `vel [1.0, 50.0]`. The arithmetic, the
experiment and the pre-existing tuning note land in the same place.

So the lag is not a regression in the filter. It dates from the day the measurement noise
was corrected upward — correctly — and the process noise was left behind.

Do not hand-tune this before the first flight: fly the configuration that was actually
tuned, then re-run CMA-ES against this bag with the along-track lag as an explicit
objective. `kf_process_noise_vel` around 8 is where that search should start.

Fixing it means another CMA-ES pass re-scored on this bag, not a hand-tweak — it was not
attempted before the first flight, and `PREFLIGHT_CHECKLIST.txt` already says to treat that
flight as the filter's first hardware validation. The raw triangulation path is healthy and
is what this measures; nothing here blocks flying.

## Every setting, in file order

Below is the whole file: every setting, in the order it appears, preceded by the prose that
used to sit above it in the YAML. **That prose no longer exists in `runtime_stack.yaml`** —
it was moved here on 2026-09-12, verbatim, when the YAML was cut from 583 lines to 206 so it
could be handed over as a file of values rather than an essay. This document is therefore
the only copy of the reasoning; the YAML keeps one short note per setting.

Nothing below was reworded in transit. Line numbers refer to the current
`config/runtime_stack.yaml`; the prose is anchored to the setting it introduced, by name, so
it stays correct when the file is reformatted.

### `launcher`

| line | setting | value | note in the file |
|---|---|---|---|
| 9 | `drone_run_mode` | `"decentralized_evaluation"` | one of: simple_detection, decentralized, decentralized_evaluation |
| 10 | `use_sim_time` | `false` | REAL DRONES: no /clock, wall time only |
| 14 | `auto_start_node` | `triangulation` |  |


### `vision`

| line | setting | value | note in the file |
|---|---|---|---|
| 17 | `publish_annotated_image` | `true` | detection boxes on /detection...; Labeled images visible on rqt. Costs bandwidth |
| 18 | `publish_debug_annotated_image` | `true` | per-observer ray overlays on /{drone}/triangulation_debug_annotated/compressed |
| 19 | `publish_detector_evaluation_packets` | `true` | false = observer emits no evaluator telemetry |
| 20 | `publish_camera_static_tf` | `true` | MUST STAY TRUE on drones: without it there is no fcu->camera TF, so no rays |

**the RealSense driver publishes its chain UN-namespaced (rgbd_link->rgbd_color_frame->**

*from the YAML comments, `model_path`*

```text
the RealSense driver publishes its chain UN-namespaced (rgbd_link->rgbd_color_frame->
rgbd_color_optical_frame, plus the depth branch) and it has NO link to fcu at all.
The only fcu->rgbd_link transforms in the bag are uav8/fcu->uav8/rgbd_link and
uav81/fcu->uav81/rgbd_link, i.e. the ones THIS node publishes. Set it false and the
TF chain from fcu to the camera does not exist, so the observer produces no rays.
It is not a bag-replay-only flag despite what the old comment claimed.
```

| line | setting | value | note in the file |
|---|---|---|---|
| 21 | `model_path` | `"dataset_tools/weights8/best_openvino_model"` | YOLO/OpenVINO weights for observer_node; relative paths are resolved from the repo root |
| 22 | `device` | `"cpu"` | inference device: "cpu" for Intel drone, "cuda" for GPU, "GPU" for Intel iGPU via OpenVINO |
| 23 | `observation_topic_name` | `"target_observation"` | per-observer frozen 3D measurement topic suffix, e.g. /uav1/target_observation |
| 24 | `allow_tf_fallback` | `true` | true = use the latest TF when the exact frame-time TF is missing |
| 25 | `max_tf_delta_ms` | `40.0` | image/TF skew budget (ms); packets exceeding it are rejected |
| 26 | `observer_image_topic` | `"/uav8/rgbd/color/image_raw/compressed"` | observer2 = uav8; camera_info derived as /uav8/rgbd/color/camera_info |
| 27 | `main_image_topic` | `"/uav11/rgbd/color/image_raw/compressed"` | observer1 = uav11; camera_info derived as /uav11/rgbd/color/camera_info |
| 28 | `observer_camera_info_topic` | `""` | role-specific CameraInfo override for observer2; empty = derive from observer_image_topic |
| 29 | `main_camera_info_topic` | `""` | role-specific CameraInfo override for observer1/main; empty = derive from main_image_topic |

**ROI logic: at margin=1.0 the node chooses a base crop whose resize to roi_reference_imgsz**

*from the YAML comments, `roi_crop_enabled`*

```text
ROI logic: at margin=1.0 the node chooses a base crop whose resize to roi_reference_imgsz
matches the full-frame target scale seen at imgsz. After margin growth, miss escalation,
and edge clamping, the node recomputes the actual YOLO imgsz from the real crop size.
```

| line | setting | value | note in the file |
|---|---|---|---|
| 30 | `roi_crop_enabled` | `true` | use ROI mode when a valid predicted center exists; false = always full-frame |
| 31 | `roi_reference_imgsz` | `256` | reference ROI side (px) defining the base crop, not the YOLO imgsz |
| 32 | `roi_margin_multiplier` | `1.40` | expands the search area around the predicted center |
| 33 | `roi_min_size_px` | `200` | 1080 \| 720: 133 \| 480: 67. Reject ROI mode if the real crop becomes smaller |

**than this in source-image pixels. The base crop is int(256*H/896)*1.40, i.e.**

*from the YAML comments, `roi_track_timeout_sec`*

```text
than this in source-image pixels. The base crop is int(256*H/896)*1.40, i.e.
287 px tall at 720 and 191 px at 480 -- the original 200 sat BETWEEN those two,
so at 480 it rejected the window on every clean-track frame and ROI never ran.
Keep this comfortably under the base crop height for whatever stream you fly.
```

| line | setting | value | note in the file |
|---|---|---|---|
| 34 | `roi_track_timeout_sec` | `0.5` | after this time without updates, drop the ROI track and reacquire globally |
| 35 | `roi_full_scan_interval` | `20` | every N frames force one full-frame scan to recover from drift or missed target |
| 36 | `roi_init_m` | `3` | start an ROI track only if at least M detections are mutually consistent ... |
| 37 | `roi_init_n` | `5` | ... within the last N candidate detections |
| 38 | `roi_init_max_spread_px` | `80` | 1080 \| 720: 53 \| 480: 27 -- px spread the N detections must fall within |
| 39 | `roi_selection_lambda` | `0.3` | penalty on detections far from the predicted center while ROI is active |
| 40 | `camera_frame_override` | `""` | override camera frame_id for TF lookups; empty = take it from CameraInfo |
| 41 | `publish_rviz_markers` | `false` | predicted trajectory & position markers to RViz (disable to save CPU) |


### `tight_coupling`

**Top-Down 3D→2D Tight Coupling (Information-Driven Feedback Architecture)**

*from the YAML comments, section `tight_coupling`*

```text
Top-Down 3D→2D Tight Coupling (Information-Driven Feedback Architecture)
When enabled, the triangulation node's 3D estimate is used to create a
"3D-Informed Gate" on each observer's image plane. Detections that pass the
local 2D Kalman gate but fall outside this 3D-projected ellipse are rejected.
Reference: Carlson, "Federated Square Root Filter for Decentralized Parallel
Processes", IEEE Trans. Aerospace & Electronic Systems, 1990.
Safety switch: feedback is automatically disabled when the 3D filter's
covariance exceeds a threshold or the estimate is too old (prevents the
"Swallowing Effect" during prolonged occlusions).

Velocity-Compensated Projection (latency correction):
The 3D estimate is forward-extrapolated from its publish time to the current
camera frame time using:  p(t_now) = p(t_odom) + v*dt + 0.5*a*dt²
This eliminates spurious TC rejections during sharp observer maneuvers
(e.g. synchronized pitch-forward/backward) where the stale world point would
otherwise project to the wrong pixel due to TF rotation mismatch.
The covariance is also inflated by (sigma_v_inflate * dt)² per axis so that
the gate widens naturally during the prediction horizon.
```

| line | setting | value | note in the file |
|---|---|---|---|
| 45 | `enabled` | `false` | IF TRIANGULATION IS DECENT GO TRUE master switch; false = pure loose coupling (identical to current behavior) |
| 46 | `target_odometry_topic` | `"/target/odometry_estimate"` | topic published by triangulation_node |
| 47 | `max_3d_age_sec` | `0.5` | ignore 3D state older than this (stale data = no feedback) |
| 48 | `max_3d_covariance_trace` | `100.0` | if tr(P_pos) > this, the 3D estimate is too uncertain → disable feedback |
| 49 | `gating_chi2_threshold` | `9.21` | chi² threshold for 2-DOF gate (99% confidence interval for 2 DOF) |
| 50 | `sigma_v_inflate` | `1.5` | [m/s] velocity uncertainty gate inflation over dt (use 1-2) |

**Gate grows by (sigma_v_inflate * dt)² per axis. Raise this if TC still**

*from the YAML comments, `r_pixel_sigma_px`*

```text
Gate grows by (sigma_v_inflate * dt)² per axis. Raise this if TC still
rejects valid detections during aggressive observer maneuvers.
```

| line | setting | value | note in the file |
|---|---|---|---|
| 51 | `r_pixel_sigma_px` | `10.0` | 1080 \| 720: 6.7 \| 480: 3.3 -- px, pinhole noise floor |


### `camera_tf`

**TC gate covariance: R_pixel = diag(sigma²,sigma²). Should match kf_sigma_px**

*from the YAML comments, section `camera_tf`*

```text
TC gate covariance: R_pixel = diag(sigma²,sigma²). Should match kf_sigma_px
so the local 2D KF and the 3D TC gate share the same noise model.
Static TF: {uav_name}/fcu rgbd_link (camera mount on each drone)
Measure x/y/z offset in metres from FCU center to camera center (body FLU: x=fwd, y=left, z=up)
```

| line | setting | value | note in the file |
|---|---|---|---|
| 57 | &nbsp;&nbsp;`x` | `0.085` | metres, camera offset from fcu |
| 58 | &nbsp;&nbsp;`y` | `0.0` |  |
| 59 | &nbsp;&nbsp;`z` | `-0.085` |  |
| 60 | &nbsp;&nbsp;`roll` | `0.0` | radians |
| 61 | &nbsp;&nbsp;`pitch` | `0.0` | PROVISIONAL. Do not tune here: the bias is per-boot, see docs/config_rationale.md |
| 62 | &nbsp;&nbsp;`yaw` | `0.0` | 0.0: origin frames are nord-aligned, no camera yaw exists |
| 63 | &nbsp;&nbsp;`yaw_offset` | `0.0` | [GENERAL] world-origin heading alignment (radians) |
| 66 | &nbsp;&nbsp;`x` | `0.085` | metres, camera offset from fcu |
| 67 | &nbsp;&nbsp;`y` | `0.0` |  |
| 68 | &nbsp;&nbsp;`z` | `-0.085` |  |
| 69 | &nbsp;&nbsp;`roll` | `0.0` | radians |
| 70 | &nbsp;&nbsp;`pitch` | `0.0` | PROVISIONAL. Do not tune here: the bias is per-boot, see docs/config_rationale.md |
| 71 | &nbsp;&nbsp;`yaw` | `0.0` | 0.0: origin frames are nord-aligned, no camera yaw exists |
| 72 | &nbsp;&nbsp;`yaw_offset` | `0.0` | [GENERAL] world-origin heading alignment (radians) |

**but the value is and has always been 0.0. Measure an offset and write it here if you**

*from the YAML comments, `child_frame`*

```text
but the value is and has always been 0.0. Measure an offset and write it here if you
need one; do not assume one is already in effect.
```

| line | setting | value | note in the file |
|---|---|---|---|
| 73 | `child_frame` | `"rgbd_link"` | camera driver frame name (same on all drones) |


### `(top level)`

**inspect the mount live:  ros2 run tf2_ros tf2_echo uav11/world_origin uav11/fcu**

*from the YAML comments, `bag_profile`*

```text
inspect the mount live:  ros2 run tf2_ros tf2_echo uav11/world_origin uav11/fcu
---------------------------------------------------------------------------
PER-FLIGHT SURVEY CONSTANTS.  Empty on purpose in the DEPLOY tree.

The bearing_bias numbers that live here in the bagtest tree are a fit to ONE
BOOT of one flight: the level reference is set at each power-up, so uav81's
-9.07 deg ground pitch belongs to that power-up and to no other.  Carrying a
profile forward injects 4-8 deg of pointing error, which at 18 m is 0.31 m per
degree.  On a real flight there is also no target ground truth to fit against.

So: keep this "none".  If a flight is recorded and later analysed offline, fit
the bias in the bagtest tree, on that bag, and leave this file alone.
```

| line | setting | value | note in the file |
|---|---|---|---|
| 76 | `bag_profile` | `"none"` | "none" WHEN FLYING |


### `bag_profiles`

| line | setting | value | note in the file |
|---|---|---|---|
| 80 | `none` | `{}` | no per-flight correction -- the only valid value for a real flight |

**NOT for deploy: bag_profile above stays "none" and this block is reachable**

*from the YAML comments, section `temesvar_A`*

```text
NOT for deploy: bag_profile above stays "none" and this block is reachable
only via BAG_PROFILE=temesvar_A on an offline replay. Fitted to ONE BOOT of
the 2026-08-29 17:21 flight; carrying it to another flight injects 4-9 deg
of pointing error. The fix belongs on the airframe, not here.
```

**Fit on 933 uav11 / 910 uav8 rays against /uav2/estimation_manager/odom_main,**

*from the YAML comments, section `bearing_bias`*

```text
Fit on 933 uav11 / 910 uav8 rays against /uav2/estimation_manager/odom_main,
model el_error = tilt + dz/range (separable because range spans 9-55 m),
robust IRLS so the mis-associations do not drag it. The two tilts have
OPPOSITE sign: 9.16 deg of differential is what opens the rays.
Held out: coefficients fitted on the first half, applied to the second ->
ray gap 3.45 -> 0.41 m, gate acceptance 29 -> 100%. Ray gap is GT-free.
```

| line | setting | value | note in the file |
|---|---|---|---|
| 87 | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`az_deg` | `0.57` | bag replay only (deg) |
| 88 | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`el_deg` | `-4.23` | bag replay only (deg) |
| 91 | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`az_deg` | `0.80` | bag replay only (deg) |
| 92 | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`el_deg` | `4.92` | bag replay only (deg) |

**Common term scaling as 1/range, so it is metric, not angular: the datum.**

*from the YAML comments, `target_gt_z_offset_m`*

```text
Common term scaling as 1/range, so it is metric, not angular: the datum.
It moves the GT, not the rays -- it fixes the yardstick, not the estimate.
Which side is wrong (uav2's z or the observers') is not decidable from the
bags: uav2 publishes no /tf. -0.90 m is what zeroes the 1/range term.
```

| line | setting | value | note in the file |
|---|---|---|---|
| 93 | &nbsp;&nbsp;`target_gt_z_offset_m` | `-0.90` | bag replay only (m) |


### `simulation`

| line | setting | value | note in the file |
|---|---|---|---|
| 96 | `drone_count` | `3` | allowed values: 1, 2, 3 |


### `uav_roles`

| line | setting | value | note in the file |
|---|---|---|---|
| 99 | `observer1` | `uav11` | main/triangulating drone (22-7 bag) |
| 100 | `observer2` | `uav8` | secondary observer drone (22-7 bag) |
| 101 | `target` | `uav2` | TARGET drone name edit to match your real drone |


### `(top level)`

**Shared heading (STATIC BENCH ONLY -- leave false to fly)**

*from the YAML comments, `shared_heading`*

```text
Shared heading (STATIC BENCH ONLY -- leave false to fly)
---------------------------------------------------------------------------
Rotates uav11's rays so they are built with uav8's yaw instead of their own, live, every
frame. What it buys: the two compasses stop disagreeing, and it is the DIFFERENTIAL
azimuth error that inflates range as R^2/base (8.1 m/deg at 50 m on a 5.4 m base) while
the common-mode error that survives only rotates the solution (0.87 m/deg). On the v5 bag
the two magnetometers sat parallel on a table and disagreed by 17.7 deg.

It assumes the two airframes' TRUE headings are equal. That holds only while they sit
parallel on the ground. Airborne, any real heading difference goes straight into uav11's
bearing as an error -- set this back to false before flying. The node prints a WARN at
startup for as long as it is on, and logs the live delta every 5 s.

Only yaw is shared: roll/pitch come from the IMU, not the compass, and two hovering
drones genuinely differ there. Composes additively with bag_profiles.*.bearing_bias
(both are rotations about world Z), so it is safe alongside bag_profile: none.
Forces the observer odom subscriptions on even with use_odom_uncertainty: false;
staleness uses odom_max_age_sec and a stale frame falls back to uav11's own heading.
```

| line | setting | value | note in the file |
|---|---|---|---|
| 102 | `shared_heading` | `false` | STATIC BENCH ONLY -- ALWAYS FALSE EXCEPT FOR TESTING |


### `ekf`

| line | setting | value | note in the file |
|---|---|---|---|
| 106 | `kf_process_noise_acc` | `13.28` | CA jerk std-dev (m/s^3) |
| 107 | `kf_process_noise_vel` | `2.0` | CV acceleration std-dev (m/s^2) |

**acceleration a quadrotor can have: on bag A the target reached**

*from the YAML comments, `imm_p_stay_cv`*

```text
acceleration a quadrotor can have: on bag A the target reached
7.8 m/s. With Q that small against R ~1.7 m² at 30 m the filter
moved 4% of the way toward each measurement (measured gain 0.04),
averaging over ~1.7 s and lagging fast motion by ~0.75 s.
---------------------------------------------------------------------------
IMM Markov Transition Matrix
Controls how quickly the filter switches between the two dynamic models:
  CV (Constant Velocity): smooth, biased for straight/hover flight
  CA (Constant Acceleration): reactive, active during braking and sharp turns

The 2x2 Markov matrix is:
  [ imm_p_stay_cv,       1 - imm_p_stay_cv   ]   (CV row)
  [ imm_p_return_to_cv,  1 - imm_p_return_to_cv ] (CA row)

imm_p_stay_cv — P(CV→CV): probability of remaining in straight-flight mode.
  Range: [0.60 – 0.95]
  ↓ Lower  → CA activates faster at braking onset (more reactive on curves)
  ↑ Higher → CA activates slower, filter prefers smooth straight-line prediction
  Recommended for aggressive maneuvering targets: 0.70–0.80
  Too low (<0.60): jittery estimation during straight flight (ghosting acceleration)
```

| line | setting | value | note in the file |
|---|---|---|---|
| 108 | `imm_p_stay_cv` | `0.75` | P(CV->CV). Range 0.60-0.95; lower = CA reacts sooner to braking |

**imm_p_return_to_cv — P(CA→CV): probability that CA model decays back to CV.**

*from the YAML comments, `imm_p_return_to_cv`*

```text
imm_p_return_to_cv — P(CA→CV): probability that CA model decays back to CV.
  Range: [0.15 – 0.50]
  ↑ Higher → CA fades faster after a maneuver (cleaner steady-state)
  ↓ Lower  → CA persists longer through long tight turns
  Recommended: 0.25–0.35 for UAV targets with short aggressive maneuvers
  Too low (<0.15): filter "locks on" to ghost acceleration after curves end
```

| line | setting | value | note in the file |
|---|---|---|---|
| 109 | `imm_p_return_to_cv` | `0.30` | P(CA->CV). Range 0.15-0.50; higher = CA fades faster after a manoeuvre |
| 110 | `kf_measurement_noise` | `0.0019` | visual angular variance (rad^2) = sigma 2.5 deg; MUST match tuning.yaml |

**Used as V_vis = kf_measurement_noise * d^2 * I (isotropic bearing cone)**

*from the YAML comments, section `anisotropic_bearing_noise`*

```text
Used as V_vis = kf_measurement_noise * d^2 * I (isotropic bearing cone)
whenever anisotropic_bearing_noise.enabled is false; ignored when true.
MUST match tuning.yaml fixed_kf_measurement_noise.
Sized from the FULL budget, not the detector: the camera is bolted to fcu
with identity rotation, so attitude error enters the bearing 1:1 and
dominates it (1.5-3.9 deg measured). A detector-only 0.0002 claimed
sigma_range 3.8 m at 49 m against a true 16.4 m -> the measured ANEES ~64.
WARNING: kf_process_noise_* and kf_mahalanobis_threshold were CMA-ES-tuned
against the old 0.0002, so the gate is looser than it was tuned to be.
---------------------------------------------------------------------------
Anisotropic Bearing Noise  (elliptical bearing cone)
---------------------------------------------------------------------------
kf_measurement_noise above is a SINGLE angular variance: it models the ray
uncertainty as a CIRCULAR cone, identical in azimuth and elevation. On this
airframe that is not what the error budget looks like. The camera is bolted
to the FCU with identity rotation, so attitude error enters the bearing 1:1,
and the two attitude axes have completely different sources:

  azimuth   (yaw-like)  <- dual-antenna RTK heading. sigma ~ sigma_rel / L;
                           with L = 0.55 m and sigma_rel ~ 1 cm this is ~1.0 deg.
                           Rotation about the antenna baseline is unobservable.
  elevation (pitch-like) <- per-boot level reference latched at power-up.
                           Measured 1.5-3.9 deg on the 22-7 bags (uav81 sat at
                           -9.07 deg pitch bias on the ground before calibration).

With enabled=true the visual term of V_i becomes, on the ray's own triad,

  V_vis = d_i^2 * diag( sigma_az^2 * cos^2(el),  sigma_el^2,  radial filler )

where the triad is (a, e, d) with a = normalise(z_world x d) the horizontal
azimuth axis and e = d x a the elevation axis. The cos^2(el) factor is the
Jacobian of the spherical parametrisation (a yaw error d_az displaces the
target by r*cos(el)*d_az, not r*d_az). The radial entry never reaches R_out:
it is cancelled exactly by the outer projectors Pi_i in the Fisher sum, and
exists only to keep V_i invertible.

These two sigmas are the TOTAL per-axis budget of the ray, not the attitude
part alone. Detector jitter and attitude error are independent, so they add
in quadrature, and the detector term (~0.81 deg, i.e. the 0.00020 rad^2 that
kf_measurement_noise used to hold before it was inflated) is NOT negligible
against the RTK yaw term:

  sigma_az = sqrt(yaw_rtk^2  + detector^2) = sqrt(1.04^2 + 0.81^2) = 1.32 deg
  sigma_el = sqrt(tilt_boot^2 + detector^2) = sqrt(2.50^2 + 0.81^2) = 2.63 deg

Re-derive both if you re-measure either term (new antenna baseline, ground
calibration procedure, retrained detector).

enabled: false -> kf_measurement_noise is used isotropically (previous
                  behaviour, bit-for-bit identical). Sigmas below ignored.
enabled: true  -> kf_measurement_noise is NOT used for the visual term; the
                  two sigmas below define it completely.

What it buys, measured on the 9.3 m baseline at 49 m range: the vertical axis
of R_out is unchanged (1.52 -> 1.60 m, the elevation budget really is 2.5 deg)
while the RANGE axis halves (16.07 -> 8.48 m). Azimuth error is amplified into
range by r^2/B ~ 258 m/rad; elevation error is not amplified at all. The
isotropic 2.5 deg was paying the pitch-bias budget on the axis governed by the
much better RTK yaw. That factor 1.90 is the whole point of this block.

Validity: the az/el split is EXACT for a target at the observer's altitude.
The only cross-term is a tilt error along the line of sight leaking into
azimuth with factor sin(el) -- 8.7% at el=5 deg, 34% at el=20 deg. If you
start flying well above or below the target, raise sigma_az accordingly.

Caveat: the per-boot tilt is a BIAS, not white noise. Putting it in R stops
the filter being instantly overconfident, but it does not stop P from
shrinking below the bias floor over a long lock. The real fix is estimating
it (BIASFIT) or augmenting the state; this block is the honest-R fallback.
```

| line | setting | value | note in the file |
|---|---|---|---|
| 112 | &nbsp;&nbsp;`enabled` | `false` | master switch; false = exact previous isotropic behaviour. The true path is STILL NOT TESTED |
| 113 | &nbsp;&nbsp;`sigma_az_deg` | `1.32` | total azimuth budget [deg] (RTK yaw + detector) |
| 114 | &nbsp;&nbsp;`sigma_el_deg` | `2.63` | total elevation budget [deg] (boot tilt + detector) |
| 117 | `kf_adaptive_cov_max_multiplier` | `50.0` | INERT: read but never used, see docs/config_rationale.md |
| 118 | `kf_initial_covariance` | `2.0` | initial position variance (m^2) |
| 119 | `kf_mahalanobis_threshold` | `34.0` | chi2(3, 0.999). Raised deliberately -- read the doc before changing |

**If this is too small, the filter will reject valid braking frames and coast 10m!**

*from the YAML comments, `ml_refine`*

```text
If this is too small, the filter will reject valid braking frames and coast 10m!
---------------------------------------------------------------------------
ML refinement (Gauss-Newton on the angular residual)
The WLS solve above is PSEUDOLINEAR: it minimises a metric residual, which is
what makes it closed-form, and also what makes it biased -- the bias does not
vanish as measurements accumulate (Gavish & Weiss 1992, Sec. III). This flag
adds 2-4 Gauss-Newton steps on the true angular residual, started from the WLS
answer: asymptotically unbiased and CRLB-efficient (ibid., Sec. IV).
Cost: ~2-4 3x3 solves per frame, negligible at 15 Hz.
Falls back to the WLS answer if a step diverges, so it cannot be worse.
Weights come from anisotropic_bearing_noise when that is enabled, otherwise
from kf_measurement_noise on both axes.

A/B on a bag: flip this, relaunch, and compare the MEDIAN radial error of
/target/triangulation_raw against /target/ground_truth_aligned. See
refine_triangulation_gn() for why the median and not the mean.
```

| line | setting | value | note in the file |
|---|---|---|---|
| 120 | `ml_refine` | `false` | master switch; false = exact previous WLS-only behaviour |
| 121 | `ml_refine_iters` | `5` | max Gauss-Newton iterations (stops early at 0.1 mm step) |

**normalize_crlb - how R from triangulate_wls reaches the IMM update.**

*from the YAML comments, `normalize_crlb`*

```text
normalize_crlb - how R from triangulate_wls reaches the IMM update.
  false (ACTIVE, SOTA): R = inv(Lambda) raw. The Kalman gain adapts to real geometry:
                        large at long range, small near a 90 deg intersection.
  true  (legacy):       same anisotropic shape, but rescaled to trace(R) = 3*kf_measurement_noise.
The flag IS honoured by triangulate_wls (the branch was once deleted and has been restored).
Requires use_odom_uncertainty=false to match the tuning config.
Re-running CMA-ES under false needs: kf_measurement_noise fixed (not optimised), wider Q bounds
(vel [1.0, 50.0], acc [1.0, 100.0]) and kf_mahalanobis_threshold re-tuned.
```

| line | setting | value | note in the file |
|---|---|---|---|
| 122 | `normalize_crlb` | `false` | false = R keeps its raw CRLB scale. NOT a no-op |


### `triangulation`

| line | setting | value | note in the file |
|---|---|---|---|
| 126 | `max_pair_age_sec` | `0.20` | maximum timestamp separation accepted for one observer1/observer2 pair |
| 127 | `observation_buffer_sec` | `1.0` | keep this much recent observation history available for pairing and debug overlays |
| 128 | `triangulation_min_baseline_m` | `0.25` | reject pairs whose camera centers are too close for stable geometry |
| 129 | `max_triangulation_error_m` | `2.0` | reject midpoint solutions when the two rays remain too far apart |
| 130 | `processing_rate_hz` | `100.0` | node timer rate for pruning buffers, coast prediction, and debug housekeeping |
| 131 | `coast_after_sec` | `0.15` | how long the filter may go unfed before the timer coasts (s) |

**Measured against the WALLCLOCK of the last accepted update,**

*from the YAML comments, `max_coast_sec`*

```text
Measured against the WALLCLOCK of the last accepted update,
not against the image capture stamp, so it is a real starvation
timeout and not a latency measurement. Tie it to the DETECTION
rate: ~1.5x the detector period (10 fps on CPU -> 0.15 s).
Lower it only if the ~10 Hz output rate is too coarse, and know
that below one detection period fusion_mode reverts to reporting
COAST over a healthy STEREO track.
```

| line | setting | value | note in the file |
|---|---|---|---|
| 132 | `max_coast_sec` | `1.0` | hard horizon for open-loop extrapolation (s) |

**measurement. Past it the node stays silent instead of publishing a**

*from the YAML comments, `publish_evaluation_packets`*

```text
measurement. Past it the node stays silent instead of publishing a
guess; it was hardcoded at 5.0 s inside _coast_predict. On bag A
(Temesvar 2026-08-29) 62% of the published output was dead reckoning
older than 1 s and 24% older than 3 s, with median error 1.76 m and
3.12 m against 0.58 m for the triangulations themselves. That
extrapolation, not the geometry, was the error.
```

| line | setting | value | note in the file |
|---|---|---|---|
| 133 | `publish_evaluation_packets` | `true` | set to true to publish /target/triangulation_evaluation for CMA-ES |

**Prediction-Guided Ray Pairing (Global Nearest Neighbor association)**

*from the YAML comments, `pairing_prediction_enabled`*

```text
Prediction-Guided Ray Pairing (Global Nearest Neighbor association)
Selects the observer1/observer2 ray pair minimizing:

  J = w_time * (|dt| / max_pair_age_sec)
    + w_geom * (e / max_triangulation_error_m)
    + w_innov * (d2 / pairing_mahalanobis_chi2)

where:
  dt  = |stamp_obs1 - stamp_obs2|
  e   = 3D ray intersection gap from triangulate_midpoint()
  d2  = (p_mid - x_pred)^T * S^{-1} * (p_mid - x_pred)
  S   = P_pred_pos + R_3d(Fisher information from both rays)

When pairing_prediction_enabled=false → legacy behavior: pick min |dt| only.
References: Bar-Shalom et al. (gating); Blackman & Popoli (GNN association).
```

| line | setting | value | note in the file |
|---|---|---|---|
| 135 | `pairing_prediction_enabled` | `true` | false = pair by closest timestamp only |
| 136 | `pairing_weight_time` | `4.0` | w_t: temporal synchronization weight |
| 137 | `pairing_weight_geom` | `1.5` | w_g: ray mismatch penalty (geometry quality) |
| 138 | `pairing_weight_innov` | `1.5` | w_i: 3D innovation consistency with IMM prediction |
| 139 | `pairing_mahalanobis_chi2` | `18.0` | NORMALISES the pairing cost; it is NOT a gate |

**It is NOT a gate and need not match ekf.kf_mahalanobis_threshold (34.0),**

*from the YAML comments, `pairing_log_best_cost`*

```text
It is NOT a gate and need not match ekf.kf_mahalanobis_threshold (34.0),
which is the real rejection gate applied inside the IMM update.
```

| line | setting | value | note in the file |
|---|---|---|---|
| 140 | `pairing_log_best_cost` | `false` | debug: log winning J, c_t, c_g, c_i at INFO throttle |

**Single-ray RGB-D Depth Fusion Fallback**

*from the YAML comments, `single_obs_max_age_sec`*

```text
Single-ray RGB-D Depth Fusion Fallback
When stereo pairing fails (e.g. mutual occlusion) but one drone has a valid
depth measurement, the system updates the IMM using a single-sensor
formulation. The 3D uncertainty is highly anisotropic: narrow radially (depth)
but wider laterally (angular).
```

| line | setting | value | note in the file |
|---|---|---|---|
| 141 | `single_obs_max_age_sec` | `0.15` | max age for an unpaired observation to be fused |
| 142 | `single_obs_min_confidence` | `0.60` | min YOLO confidence required to use single-sensor fallback |


### `kalman_filter`

**Image-space Kalman filter — false positive rejection layer inside each ObserverNode.**

*from the YAML comments, section `kalman_filter`*

```text
Image-space Kalman filter — false positive rejection layer inside each ObserverNode.
```

| line | setting | value | note in the file |
|---|---|---|---|
| 148 | `kf_enabled` | `true` | false = every raw detection becomes a ray, no false-positive layer |
| 149 | `kf_sigma_a` | `350.0` | 1080 \| 720: 200.0 \| 480: 100.0 -- px/s^2, scales with k |

**one_drone e in tuning.yaml (locked_2d_params); il 400 della vecchia**

*from the YAML comments, `kf_sigma_px`*

```text
one_drone e in tuning.yaml (locked_2d_params); il 400 della vecchia
tabella era solo la scalatura ingenua da 720, mai validato.
RESOLUTION-COUPLED KNOBS. These are absolute pixel quantities: they only mean what they
say at the resolution they were tuned at, which was 1920x1080. Active values below are for
1920x1080 (k = 1); the trailing comment carries 1280x720 (k = 2/3) and 640x480 (k = 1/3), so
switching stream resolution is a copy from the comment.
These knobs live in observer_node.py and only bite when the detector runs on a live camera.
A bag replay does NOT go through them: replay_bag_a.sh feeds triangulation_node the
/uavN/target_observation rays already computed in flight, so the replay score is unaffected.
Measured, not assumed: the Temesvar August bags carry camera_info 640x480 on both observers,
i.e. those flights ran the 720 column on 480 imagery -- gates 2x too wide, areas 4x.
  lengths (px, px/s^2) scale by k    -- kf_sigma_px, kf_sigma_a, kf_ghost_reentry_spread_px,
                                        roi_init_max_spread_px, tight_coupling.r_pixel_sigma_px
  areas / variances (px^2) by k^2    -- kf_p_max, kf_bbox_area_ref
NOT covered here: _ghost_reentry_anchor_radius_px() in observer_node.py hardcodes 48 px and a
320 px cap. The cap dominates at every resolution, so that recovery gate is relatively more
permissive the smaller the frame. Left alone on purpose -- it is the re-acquisition path and
generous is the intent -- but it is the one pixel constant this table does not reach.
```

| line | setting | value | note in the file |
|---|---|---|---|
| 150 | `kf_sigma_px` | `11.0` | 1080 \| 720: 7.3 \| 480: 3.7 |
| 151 | `kf_conf_beta` | `1.0` | weight of YOLO confidence in the measurement noise |
| 152 | `kf_conf_min` | `0.2` | minimum YOLO confidence accepted |
| 153 | `kf_n_confirm` | `3.0` | consecutive detections needed before a ray is published |
| 154 | `kf_max_age` | `5` | frames a track survives without a detection |
| 155 | `kf_p_max` | `800.0` | 1080 \| 720: 356.0 \| 480: 89.0  -- varianza px^2, scala con k^2 |
| 156 | `kf_nis_window` | `15` | samples averaged for the NIS health check |
| 157 | `kf_nis_threshold` | `8.0` | per-detection NIS gate |
| 158 | `kf_ghost_timeout_sec` | `0.5` | s before a lost track is treated as a ghost |
| 159 | `kf_ghost_reentry_m` | `2` | M of N detections needed to revive a ghost track |
| 160 | `kf_ghost_reentry_n` | `3` | ... out of the last N candidates |
| 161 | `kf_ghost_reentry_conf_min` | `0.35` | minimum confidence for a ghost re-entry |
| 162 | `kf_ghost_reentry_spread_px` | `100.0` | 1080 \| 720: 67.0 \| 480: 33.0 |
| 163 | `kf_ego_motion_enabled` | `true` | compensate the observer own motion between frames |
| 164 | `kf_bbox_area_ref` | `3600.0` | 1080 \| 720: 1600.0 \| 480: 400.0  -- area px^2, scala con k^2 |
| 165 | `kf_dist_gamma` | `0.2` | distance exponent for the confidence model |
| 166 | `kf_cam_aware_gating_enabled` | `true` | scale the 2D gate with apparent target size |


### `evaluation`

| line | setting | value | note in the file |
|---|---|---|---|
| 169 | `odom_topic_suffix` | `"estimation_manager/odom_main"` | CORRECT for real drones; hw_api/ground_truth is sim-only |
| 170 | `world_frame_suffix` | `"world_origin"` | real drone RTK origin, V2 uses f9p_origin |

**Set true ONLY when every drone is configured with the same world origin (x, y AND z)**

*from the YAML comments, `shared_world_origin`*

```text
Set true ONLY when every drone is configured with the same world origin (x, y AND z)
in the world file. The triangulation node then treats all <uav>/world_origin frames as
one frame and skips the utm datum hop in _datum_rt, which is the only thing that needs
/tf_static to cross the network between drones.
With false and no /tf_static, _transform_observation_to_target_frame returns None for
every remote observation and the second observer contributes NOTHING: no stereo pair,
no triangulation. With true and origins that do NOT actually coincide, the offset
becomes a silent constant bias on every ray. Measure before trusting: park the drones
on the same flat ground and compare the z of their estimation_manager/odom_main
(preflight_check.py --live prints exactly this).
```

| line | setting | value | note in the file |
|---|---|---|---|
| 171 | `shared_world_origin` | `true` | true = all drones share one world origin. VERIFY THE Z DATUM before trusting |

**GT re-published by the triangulation node after the utm datum hop, in the same frame as**

*from the YAML comments, `gt_aligned_topic`*

```text
GT re-published by the triangulation node after the utm datum hop, in the same frame as
/target/odometry_estimate. The raw <target>/odom topic is in the TARGET drone's own origin
frame, whose z datum differs by several metres (5.4 m in v5), so comparing it against the
estimate charts that constant gap as tracking error. Set to "" to go back to the raw topic.
```

| line | setting | value | note in the file |
|---|---|---|---|
| 172 | `gt_aligned_topic` | `"/target/ground_truth_aligned"` | GT republished by the triangulation node; "" = use the raw topic |

**Live Monitor Display Mode**

*from the YAML comments, `live_monitor_mode`*

```text
Live Monitor Display Mode
Controls which plots are shown when ./launch_live_monitor.sh or
./launch_sim.sh opens the live monitor window.

Modes:
  "classic"     — 3 panels: XY bird's-eye, Z altitude, 3D position error norm.
                  The original layout; lightest on CPU.

  "full_3d"     — 4 panels in a 2×2 grid:
                  [XY bird's-eye]  [3D trajectory vs GT]
                  [Z altitude]     [3D position error norm]
                  The 3D panel uses the same colour scheme / legend as the
                  XY panel and supports mouse rotation + scroll-to-zoom.

  "minimal_3d"  — 3 panels focused on the 3D view:
                  [3D trajectory vs GT] (large, interactive)
                  [XY error over time]  [Z error over time]
                  Best for understanding spatial trajectory divergence
                  without the bird's-eye projection ambiguity.
```

| line | setting | value | note in the file |
|---|---|---|---|
| 173 | `live_monitor_mode` | `"classic"` | one of: classic \| full_3d \| minimal_3d |
| 176 | `auto_start_with_launch_sim` | `true` |  |


### `depth_fusion`

**RGB-D Depth Fusion  (Bar-Shalom et al. — gating & scalar measurement fusion;**

*from the YAML comments, section `depth_fusion`*

```text
RGB-D Depth Fusion  (Bar-Shalom et al. — gating & scalar measurement fusion;
                     Hartley & Zisserman — projection / depth along ray;
                     Khoshelham & Elberink 2012 — depth noise vs distance)
Master switch: when enabled=false no depth subscriber is created and every
downstream depth code-path is dead — identical CPU/network load to bearing-only.
```

**Master switch. false = no depth subscriber, no depth code path, identical to bearing-only.**

*from the YAML comments, `enabled`*

```text
Master switch. false = no depth subscriber, no depth code path, identical to bearing-only.
FALSE since the USB-C 3.0 cable was swapped for USB 2.1 (RTK interference): that link has no
bandwidth for an aligned depth stream, so there is no /rgbd/aligned_depth_to_color to subscribe
to. Leaving it true makes preflight_check.py FAIL and buys nothing. Cost of turning it off:
triangulation_node's single-observation fallback (line ~3124) requires depth_valid, so with one
observer occluded the filter now only coasts -- there is no single-ray range update any more.
```

| line | setting | value | note in the file |
|---|---|---|---|
| 181 | `enabled` | `false` | false = no depth subscriber at all (USB 2.1 cannot carry the stream) |

**Topic suffix relative to uav_name: full topic = /{uav_name}/{depth_topic_suffix}**

*from the YAML comments, `depth_topic_suffix`*

```text
Topic suffix relative to uav_name: full topic = /{uav_name}/{depth_topic_suffix}
Supports sensor_msgs/Image with encoding 16UC1 (millimetres) or 32FC1 (metres).
```

| line | setting | value | note in the file |
|---|---|---|---|
| 182 | `depth_topic_suffix` | `"rgbd/aligned_depth_to_color/image_raw"` |  |

**Maximum time skew between depth frame and RGB frame (milliseconds).**

*from the YAML comments, `depth_max_align_ms`*

```text
Maximum time skew between depth frame and RGB frame (milliseconds).
Uses the same spirit as vision.max_tf_delta_ms.
```

| line | setting | value | note in the file |
|---|---|---|---|
| 183 | `depth_max_align_ms` | `40.0` | max depth/RGB time skew (ms) |

**Valid sensor range (metres). Tune from bag statistics for your hardware.**

*from the YAML comments, `min_range_m`*

```text
Valid sensor range (metres). Tune from bag statistics for your hardware.
```

| line | setting | value | note in the file |
|---|---|---|---|
| 184 | `min_range_m` | `0.3` | valid sensor range (m) |
| 185 | `max_range_m` | `10.0` | valid sensor range (m) |

**Depth sample extraction inside the YOLO bounding box (full-frame pixel coords).**

*from the YAML comments, `sample_method`*

```text
Depth sample extraction inside the YOLO bounding box (full-frame pixel coords).
Methods: "median" (default, robust to holes & flying pixels — Khoshelham 2012),
         "center_pixel", "trimmed_mean".
```

| line | setting | value | note in the file |
|---|---|---|---|
| 186 | `sample_method` | `"median"` | one of: median \| center_pixel \| trimmed_mean |
| 187 | `trimmed_mean_fraction` | `0.2` | for trimmed_mean: discard top/bottom fraction |
| 188 | `min_valid_pixel_fraction` | `0.001` | reject depth if fewer valid pixels cover the bbox |
| 189 | `depth_sample_stride` | `2` | stride for subsampling large bboxes (2 = halve CPU for area > 4000px²) |

**Depth consistency gate vs 3D odometry (collinear occluder rejection).**

*from the YAML comments, `gating_enabled`*

```text
Depth consistency gate vs 3D odometry (collinear occluder rejection).
Rejects near occluder when |z_meas - z_pred| > gate_k * sqrt(sigma_z_meas² + sigma_z_pred²).
Reference: Bar-Shalom, Li & Kirubarajan — "Estimation with Applications to
Tracking and Navigation" (2001), Sec. 3.4 (scalar gating).
```

| line | setting | value | note in the file |
|---|---|---|---|
| 190 | `gating_enabled` | `false` | reject depth samples that disagree with the 3D estimate |
| 191 | `gate_k` | `3.0` | gate width in sigmas |
| 192 | `max_odom_age_sec` | `0.5` | same spirit as tight_coupling.max_3d_age_sec |
| 193 | `max_odom_covariance_trace` | `100.0` | disable depth gate if 3D positional uncertainty is too large |
| 194 | `gate_odom_topic` | `"/target/odometry_estimate"` | [L3-bug2] Dedicated IMM odom topic for the depth consistency |

**gate, independent of tight_coupling.enabled.**

*from the YAML comments, `sigma_z_base_m`*

```text
gate, independent of tight_coupling.enabled.
Depth measurement noise model (Khoshelham & Elberink 2012, Eq. 5):
  sigma_z = sigma_z_base + sigma_z_range_coeff * z_meas
```

| line | setting | value | note in the file |
|---|---|---|---|
| 195 | `sigma_z_base_m` | `0.02` | depth noise model: sigma_z = base + coeff * z (m) |
| 196 | `sigma_z_range_coeff` | `0.01` |  |

**Triangulation fusion policy parameters (triangulation_node).**

*from the YAML comments, `fusion_use_range_constraint`*

```text
Triangulation fusion policy parameters (triangulation_node).
When one or both observers carry a valid depth measurement, the range
along the bearing ray is used as an additional 1-D Kalman update
(H_r = d_hat^T, R_r = sigma_z^2) after the bearing-only triangulation
measurement, following Bar-Shalom et al. sequential update.
```

| line | setting | value | note in the file |
|---|---|---|---|
| 197 | `fusion_use_range_constraint` | `true` | apply an accepted depth sample as a 1-D range update |
| 198 | `fusion_range_chi2` | `9.0` | chi² threshold for per-ray depth consistency check (1-DOF) |


### `observer_odom_uncertainty`

**Observer Odometry Uncertainty Propagation**

*from the YAML comments, section `observer_odom_uncertainty`*

```text
Observer Odometry Uncertainty Propagation
When enabled, the 3x3 translational position covariance of each observer
drone is extracted from its nav_msgs/Odometry message and propagated into
Observer Odometry Uncertainty (WLS Errors-In-Variables Model)

The triangulation_node now uses an Optimal Weighted Least Squares (WLS)
estimator following the Errors-In-Variables (EIV) formulation. Each bearing
ray has a composite noise covariance:

  V_i = (kf_measurement_noise * d_i^2 * I_3)  +  (Pi_i @ P_odom_i @ Pi_i^T)

The first term is isotropic only while ekf.anisotropic_bearing_noise.enabled
is false; when enabled it becomes the elliptical cone described in that block.

where Pi_i = I - b_i*b_i^T is the orthogonal projector onto the plane
perpendicular to the bearing (bearing-null-space). The Fisher Information
Matrix Lambda = sum_i (Pi_i^T * W_i * Pi_i) is inverted to yield the
Cramér-Rao Lower Bound covariance R_out, which is passed as R_k to the IMM.

Theoretical references:
  Dogancay, K. (2015). IEEE Trans. Signal Processing 63(6). [projector Pi]
  Xu, Dogancay & Hmam (2017). Signal Processing 133. [composite V_i, UAV]
  Bishop et al. (2010). Automatica 46(3). [CRLB optimality]
  Dogancay (2005). Signal Processing 85(9). [bootstrap OLS / Orth. Vectors]

Zero-overhead policy:
  When use_odom_uncertainty=false, P_odom is ignored and V_i = var_px*d^2*I_3.
```

| line | setting | value | note in the file |
|---|---|---|---|
| 202 | `use_odom_uncertainty` | `false` | MUST match tuning.yaml, or R differs between tuning and flight |

**Bags have no P_odom → training with True gives wrong R scale.**

*from the YAML comments, `odom_fallback_variance`*

```text
Bags have no P_odom → training with True gives wrong R scale.
```

| line | setting | value | note in the file |
|---|---|---|---|
| 203 | `odom_fallback_variance` | `0.01` | m²: isotropic P_odom used when driver publishes zero cov |
| 204 | `odom_max_age_sec` | `0.5` | s:  reject stale odometry and fall back to pixel-only |
| 205 | `distance_dependent_noise` | `true` | WLS: if true, visual sigma = var_px * d^2; false = constant width beam |
| 206 | `odom_max_variance` | `1.0` | m²: diagonal clamp on P_odom diagonals; prevents barometric Z-blowup |


### `observer_self_rejection`

**Observer Self-Detection Rejection (Layer 4)**

*from the YAML comments, section `observer_self_rejection`*

```text
Observer Self-Detection Rejection (Layer 4)
Uses the known 3D odometry of the OTHER observer drone to project its
expected image-plane position and reject detections that coincide with it.
This handles intra-swarm false positives when one observer drone flies
into the line-of-sight of the other.

Gate formula (Hartley & Zisserman 2004, §6.1 pinhole projection):
  gate_px = (rejection_gate_radius_m / Zc) * max(fx, fy)
Foreground condition (optional):
  reject only when Zc_observer < Zc_target_IMM - rejection_depth_margin_m

Reference: Blackman & Popoli (1999) — Design and Analysis of Modern
  Tracking Systems, Chapter 7: negative object recognition gate.
```

| line | setting | value | note in the file |
|---|---|---|---|
| 210 | `enabled` | `true` | master switch; false = zero overhead (no subscriber created) |
| 211 | `max_odom_age_sec` | `0.3` | s: reject only when other-observer odom is fresh |
| 212 | `rejection_gate_radius_m` | `0.8` | m: physical drone body radius used to compute pixel gate |
| 213 | `rejection_depth_margin_m` | `1.5` | m: unused while rejection_only_if_closer=false; kept for future re-enable |
| 214 | `rejection_only_if_closer` | `false` | false = unconditional geometric exclusion (right for two observers flying together) |
