#!/usr/bin/env python3
"""Check that observer2's debug overlay matches against the right end of a pair.

A MatchedObservationPair is always built with observer1 on the "local" side and
observer2 on the "remote" side.  The overlay is now drawn for both drones, so
observer2 has to look up pairs by their *remote* stamp; using the local stamp
silently returns a different, still-recent pair, and the red ray drawn on uav8's
image would then belong to a different instant than the green one.  That failure
produces a plausible-looking picture, which is why it gets a test.
"""
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import triangulation_node as tn

MS = 1_000_000


class _Obs:
    def __init__(self, stamp_ns):
        self.stamp_ns = stamp_ns


class _Pair:
    """Duck-typed stand-in: the lookup only ever reads the two stamps."""

    def __init__(self, local_ms, remote_ms):
        self.local_obs = _Obs(local_ms * MS)
        self.remote_obs = _Obs(remote_ms * MS)


class _FakeNode:
    def __init__(self, pairs):
        self.debug_matched_pairs = collections.deque(pairs)


def main():
    find = tn.TriangulationNode._find_closest_matched_pair

    # Two pairs far enough apart that the two sides disagree about which is closest.
    pair_a = _Pair(local_ms=100, remote_ms=500)
    pair_b = _Pair(local_ms=400, remote_ms=800)
    node = _FakeNode([pair_a, pair_b])
    max_age = 200 * MS

    # observer1 draws at its own (local) stamp: unchanged behaviour.
    got = find(node, 400 * MS, max_age)
    assert got is pair_b, 'observer1 must still match on the local side'

    # observer2 draws at stamp 500 ms, which is pair_a's remote stamp.
    got = find(node, 500 * MS, max_age, match_remote_side=True)
    assert got is pair_a, 'observer2 must match on the remote side'

    # The whole point: the old local-side lookup would have returned pair_b here —
    # not None, just quietly the wrong pair.
    wrong = find(node, 500 * MS, max_age)
    assert wrong is pair_b, 'sanity: local-side lookup really does pick the wrong pair'
    assert wrong is not got, 'the two sides must disagree, otherwise this test proves nothing'

    # Nothing within max_age on the remote side must still yield None.
    assert find(node, 2000 * MS, max_age, match_remote_side=True) is None

    print('ok: observer2 matches on the remote side, observer1 unchanged')


if __name__ == '__main__':
    main()
