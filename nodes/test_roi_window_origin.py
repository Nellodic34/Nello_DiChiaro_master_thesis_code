#!/usr/bin/env python3
"""Guards the ROI crop origin against the stale-offset bug.

roi_x1/roi_y1 are the crop ORIGIN. Everything downstream -- the KF/TC/OSR/depth
gates, choose_best_candidate, draw_predictions, _update_roi_tracker, the
evaluation packet -- adds them back to turn crop coordinates into full-frame
coordinates. The old code computed the window, rejected it on roi_min_size_px,
then fell through to full-frame inference while LEAVING roi_x1/roi_y1 non-zero:
every bbox in that frame came out shifted by the rejected window's corner.

Nothing triggered it until the camera dropped resolution. The base crop is
int(roi_reference_imgsz * H / imgsz) * roi_margin, i.e. 432 px tall at 1080,
287 at 720 and 191 at 480 -- and the old roi_min_size_px of 200 sat between the
last two. At 480 the window was rejected on EVERY clean-track frame.

_compute_roi_window now returns None instead of a half-built window, so the
caller cannot pick up an origin belonging to a crop it did not take.

Run: python3 <dir>/test_roi_window_origin.py
     <dir> is nodes/ in the deploy tree, dataset_tools/ in the bagtest tree.
     The two copies are kept byte-identical, so `diff -q` is the alignment check.
"""
import importlib.util
import os
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location('obs', os.path.join(_HERE, 'observer_node.py'))
obs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(obs)

window = obs.ObserverNode._compute_roi_window


def node(**over):
    cfg = dict(
        roi_reference_imgsz=256,
        imgsz=896,
        roi_margin=1.40,
        roi_min_size=133,          # shipping value for 1280x720
        _roi_last_bbox=(300, 200, 340, 240),
        _roi_consecutive_misses=0,
        _round_up_to_stride=obs.ObserverNode._round_up_to_stride,
    )
    cfg.update(over)
    return types.SimpleNamespace(**cfg)


def _check_window(w, img_w, img_h, min_size):
    x1, y1, x2, y2, imgsz, _ = w
    assert 0 <= x1 < x2 <= img_w and 0 <= y1 < y2 <= img_h, w
    assert (x2 - x1) >= min_size and (y2 - y1) >= min_size, w
    assert 0 < imgsz <= 896 and imgsz % 32 == 0, imgsz


def test_the_regression_config_is_rejected_outright():
    # 640x480 with the pre-fix floor of 200: base crop is 254x191, too short. The whole
    # point of the fix is that this returns None rather than a rejected-but-populated window.
    assert window(node(roi_min_size=200), 320.0, 240.0, 640, 480) is None


def test_720p_takes_the_roi_path():
    w = window(node(), 640.0, 360.0, 1280, 720)
    assert w is not None
    _check_window(w, 1280, 720, 133)
    assert (w[2] - w[0], w[3] - w[1]) == (510, 287)
    assert w[4] == 384  # same inference scale as 480p and 1080p: the letterbox normalises it away


def test_480p_takes_the_roi_path_with_its_own_floor():
    w = window(node(roi_min_size=67), 320.0, 240.0, 640, 480)
    assert w is not None
    _check_window(w, 640, 480, 67)
    assert (w[2] - w[0], w[3] - w[1]) == (254, 191)


def test_escalation_after_a_miss_grows_the_crop():
    clean = window(node(), 640.0, 360.0, 1280, 720)
    missed = window(node(_roi_consecutive_misses=1), 640.0, 360.0, 1280, 720)
    assert missed is not None
    _check_window(missed, 1280, 720, 133)
    assert (missed[3] - missed[1]) > (clean[3] - clean[1])


def test_crop_clamped_at_the_frame_edge_still_reports_a_real_window():
    # Near the right edge the crop is truncated; if what survives is under the floor the
    # answer must be None, never a truncated window with a live origin.
    w = window(node(), 1270.0, 710.0, 1280, 720)
    if w is not None:
        _check_window(w, 1280, 720, 133)


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_'):
            fn()
            print(f'PASS  {name}')
    print('all ok')
