#!/usr/bin/env python3
"""Guards the ROI crop centre against the two-tracker divergence.

The node kept two independent estimates of where the target is in the image:
the KF state (ego-motion corrected, noise-weighted velocity) which drove
candidate SELECTION, and the ROI tracker's fixed-alpha EMA of raw finite
differences which drove the CROP. On a sharp observer pitch the KF follows the
target and the EMA does not, so the crop is aimed where the target was in the
world-fixed sense while the selection gate expects it elsewhere.

extrapolated_center() closes that gap: the crop now asks the KF. These tests
pin the three things that can silently break it -- the CONFIRMED-only rule, the
forward extrapolation actually happening, and the dt guards.

Run: python3 <dir>/test_roi_kf_center.py
     <dir> is nodes/ in the deploy tree, dataset_tools/ in the bagtest tree.
     The two copies are kept byte-identical, so `diff -q` is the alignment check.
"""
import importlib.util
import os

_spec = importlib.util.spec_from_file_location(
    'observer_node', os.path.join(os.path.dirname(__file__), 'observer_node.py')
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
Tracker = _mod.ImageSpaceKalmanTracker

S = 1_000_000_000


def _confirmed(u=300.0, v=200.0, vel_u=100.0, vel_v=0.0, t0=10 * S):
    """A CONFIRMED track moving at (vel_u, vel_v) px/s, fed clean detections at 20 Hz."""
    trk = Tracker()
    trk.initialize(u, v, t0)
    t = t0
    for i in range(1, 12):
        t = t0 + i * (S // 20)
        dt = (t - t0) / 1e9
        det = _mod.DetectionObservation(
            stamp_ns=t, center_u=u + vel_u * dt, center_v=v + vel_v * dt,
            confidence=0.9, class_id=0, class_name='drone', bbox_w=40.0, bbox_h=40.0,
        )
        trk.update(det, t)
    assert trk.state == Tracker.CONFIRMED, trk.state
    return trk, t


def test_tentative_track_gives_no_centre():
    # One hit: velocity is still the zero initial condition. Extrapolating it would
    # drag the crop backwards, so the caller must fall back to the EMA instead.
    trk = Tracker()
    trk.initialize(300.0, 200.0, 10 * S)
    assert trk.state == Tracker.TENTATIVE
    assert trk.extrapolated_center(10 * S + S // 20) is None


def test_supplied_velocity_drives_the_extrapolation():
    # The caller passes the ROI EMA velocity because the KF's own converges far too
    # slowly (see extrapolated_center docstring). Pin that the override is honoured:
    # 100 px/s over 100 ms must move the centre ~10 px, whatever the KF state says.
    trk, t = _confirmed(vel_u=100.0)
    at_now = trk.extrapolated_center(t, vel_px_s=(100.0, 0.0))
    ahead = trk.extrapolated_center(t + S // 10, vel_px_s=(100.0, 0.0))
    assert at_now is not None and ahead is not None
    assert abs((ahead[0] - at_now[0]) - 10.0) < 1e-6, (at_now, ahead)


def test_kf_own_velocity_is_the_fallback_not_the_preference():
    # Guards the regression this split exists to prevent: on a 100 px/s track the KF
    # velocity is still ~20 px/s at 0.55 s, so extrapolating with it would under-shoot
    # by 4x. The supplied EMA velocity must move the centre strictly further.
    trk, t = _confirmed(vel_u=100.0)
    with_kf = trk.extrapolated_center(t + S // 5)                          # KF velocity
    with_ema = trk.extrapolated_center(t + S // 5, vel_px_s=(100.0, 0.0))  # EMA velocity
    base = trk.predicted_center
    assert (with_ema[0] - base[0]) > 2.0 * (with_kf[0] - base[0]), (base, with_kf, with_ema)


def test_extrapolation_beats_a_frozen_centre_on_a_moving_target():
    # The whole point: a crop centred on the last posterior lags a moving target by
    # vel*dt. The extrapolated centre must land closer to where the target really is.
    vel_u, dt_s = 300.0, 0.1
    trk, t = _confirmed(u=300.0, v=200.0, vel_u=vel_u)
    posterior = trk.predicted_center
    extrapolated = trk.extrapolated_center(t + int(dt_s * S), vel_px_s=(vel_u, 0.0))
    truth_u = posterior[0] + vel_u * dt_s
    assert abs(extrapolated[0] - truth_u) < abs(posterior[0] - truth_u), (
        posterior, extrapolated, truth_u
    )


def test_stale_and_non_monotonic_stamps_are_refused():
    trk, t = _confirmed()
    assert trk.extrapolated_center(t - S // 10) is None      # stamp goes backwards
    assert trk.extrapolated_center(t + 2 * S) is None        # 2 s gap: CV model is void


def test_centre_is_clipped_into_the_frame():
    # A fast track heading off the right edge must not hand the crop builder an
    # off-frame centre: _compute_roi_window would build a window around it and the
    # clamp there would silently shift the origin.
    trk, t = _confirmed(u=500.0, v=200.0, vel_u=300.0)
    c = trk.extrapolated_center(
        t + (3 * S) // 10, image_size=(640, 480), vel_px_s=(300.0, 0.0)
    )
    assert c is not None
    assert 0.0 <= c[0] <= 639.0 and 0.0 <= c[1] <= 479.0, c


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_'):
            fn()
            print(f'PASS  {name}')
    print('all ok')
