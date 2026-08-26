#!/usr/bin/env python3
"""Guards the clock domain of the coast trigger in processing_timer_callback().

Two clocks meet in that callback and only one pairing of them is meaningful:

  imm_last_stamp_ns        image CAPTURE time, stamped on the observer drone
  now_ns                   wallclock on the triangulation node, read by the timer

Their difference is not an age, it is an age PLUS the pipeline latency (JPEG ->
WiFi -> YOLO on CPU -> WiFi). With ~10 fps detection that latency on its own
exceeds any threshold worth setting, so a guard written as
`now_ns - imm_last_stamp_ns < thresh` never holds, the 100 Hz timer falls into
the coast branch on every single tick, and /tracker/fusion_mode reports COAST
over a perfectly healthy STEREO track -- destroying the active-vs-coast
stratification the evaluation run exists to produce.

The fix is a second marker, _last_measurement_wall_ns, written on the wallclock
whenever the filter accepts a measurement, so the guard subtracts wallclock from
wallclock. _coast_predict() keeps using imm_last_stamp_ns: extrapolating capture
time to now IS latency compensation and is correct there.

This test drives the real callback through a stub node. It fails on the
pre-fix code -- that is its whole reason to exist.

Run: python3 nodes/test_coast_guard.py
"""
import importlib.util
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location('tn', os.path.join(_HERE, 'triangulation_node.py'))
tn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tn)

LATENCY_NS = 100_000_000     # 0.1 s capture -> filter, measured on the CPU detector
DETECTION_PERIOD_NS = 100_000_000   # 10 fps
COAST_AFTER_SEC = 0.15
TICK_NS = 10_000_000         # 100 Hz timer


class _Clock:
    def __init__(self):
        self.nanoseconds = 0

    def now(self):
        return self


class FakeNode:
    """Minimum surface processing_timer_callback() touches, nothing more."""

    def __init__(self):
        self._clock = _Clock()
        self.coast_after_sec = COAST_AFTER_SEC
        self.observer1_observations = []
        self.observer2_observations = []
        self.single_obs_min_confidence = 0.60
        self.single_obs_max_age_ns = 150_000_000
        self.imm_initialized = True
        self.imm_last_stamp_ns = None
        self._last_measurement_wall_ns = 0
        self.last_fusion_mode = "STEREO"
        self.coast_calls = []

    # --- clock -------------------------------------------------------------
    def get_clock(self):
        return self._clock

    def advance(self, ns):
        self._clock.nanoseconds += ns

    # --- collaborators stubbed out ----------------------------------------
    def _prune_stale_observations(self, now_ns):
        pass

    def _log_pairing_diagnostics(self, now_ns):
        pass

    def _publish_fusion_mode(self):
        pass

    def _coast_predict(self, now_ns):
        self.coast_calls.append(now_ns)

    # --- the thing under test ---------------------------------------------
    def tick(self):
        tn.TriangulationNode.processing_timer_callback(self)

    def accept_measurement(self, capture_stamp_ns):
        """Mirror of the three sites that write imm_last_stamp_ns."""
        self.imm_last_stamp_ns = capture_stamp_ns
        self._last_measurement_wall_ns = self._clock.nanoseconds
        self.last_fusion_mode = "STEREO"


def test_latency_does_not_trigger_coast():
    node = FakeNode()

    # A frame captured at t=0 reaches the filter at t=LATENCY.
    node.advance(LATENCY_NS)
    node.accept_measurement(capture_stamp_ns=0)
    accepted_at = node._clock.nanoseconds

    # 1) No coast inside the starvation window, however stale the CAPTURE stamp is.
    while node._clock.nanoseconds < accepted_at + int(COAST_AFTER_SEC * 1e9) - TICK_NS:
        node.advance(TICK_NS)
        node.tick()
    assert not node.coast_calls, (
        f'coasted {len(node.coast_calls)}x within {COAST_AFTER_SEC}s of an accepted '
        f'measurement: the guard is differencing wallclock against capture time, so it '
        f'is measuring pipeline latency ({LATENCY_NS / 1e9:g}s) instead of starvation'
    )
    assert node.last_fusion_mode == "STEREO", (
        f'fusion_mode overwritten to {node.last_fusion_mode!r} while the filter was fed'
    )
    print('  no coast within coast_after_sec of an accepted measurement  OK')

    # 2) Genuine starvation still coasts.
    while node._clock.nanoseconds < accepted_at + int(COAST_AFTER_SEC * 1e9) + 2 * TICK_NS:
        node.advance(TICK_NS)
        node.tick()
    assert node.coast_calls, 'no coast after the starvation window elapsed'
    first = (node.coast_calls[0] - accepted_at) / 1e9
    assert COAST_AFTER_SEC <= first <= COAST_AFTER_SEC + 2 * TICK_NS / 1e9, (
        f'first coast at {first:.3f}s after the update, expected ~{COAST_AFTER_SEC}s'
    )
    assert node.last_fusion_mode == "COAST", 'genuine starvation must be labelled COAST'
    print(f'  coast fires {first:.3f}s after the last update  OK')


def test_steady_state_10fps_never_coasts():
    """The flight case: detections at 10 fps, each 0.1 s late. Must never coast."""
    node = FakeNode()
    capture = 0
    node.advance(LATENCY_NS)

    for _ in range(50):                      # 5 s of flight
        node.accept_measurement(capture)
        for _ in range(DETECTION_PERIOD_NS // TICK_NS):
            node.advance(TICK_NS)
            node.tick()
        capture += DETECTION_PERIOD_NS

    assert not node.coast_calls, (
        f'coasted {len(node.coast_calls)}x during 5 s of uninterrupted 10 fps detection'
    )
    assert node.last_fusion_mode == "STEREO"
    print('  5 s at 10 fps with 0.1 s latency: 0 coasts, mode stays STEREO  OK')


def main():
    test_latency_does_not_trigger_coast()
    test_steady_state_10fps_never_coasts()
    print('OK')


if __name__ == '__main__':
    main()
