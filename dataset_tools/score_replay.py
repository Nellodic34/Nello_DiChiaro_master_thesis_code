#!/usr/bin/env python3
"""Score a recorded replay of the triangulation stack against its own aligned GT.

Record these three topics while replaying a bag, then point this at the result:

    ros2 bag record -o /tmp/rec_run \
        /target/odometry_estimate /target/triangulation_raw /target/ground_truth_aligned
    python3 dataset_tools/score_replay.py /tmp/rec_run

The point of recording triangulation_raw alongside the estimate is that it
separates the two halves of the stack.  If the measurements are good and the
estimate is not, the geometry is fine and the filter is not - which is exactly
what bag A (Temesvar 2026-08-29) turned out to be: 0.58 m in, 1.92 m out.

'guadagno' is the effective filter gain: how far the estimate moves toward each
measurement, measured at identical timestamps.  1.0 copies the measurement, 0.0
ignores it.  Anything near zero means the filter distrusts its own sensor.
"""
import sys

import numpy as np
import rosbag2_py
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry
from rclpy.serialization import deserialize_message

TYPES = {'nav_msgs/msg/Odometry': Odometry, 'geometry_msgs/msg/PointStamped': PointStamped}
EST = '/target/odometry_estimate'
RAW = '/target/triangulation_raw'
GT = '/target/ground_truth_aligned'


def read(uri):
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=uri, storage_id='mcap'),
                rosbag2_py.ConverterOptions('', ''))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    out = {}
    while reader.has_next():
        topic, data, _ = reader.read_next()
        cls = TYPES.get(types.get(topic))
        if cls is None:
            continue
        msg = deserialize_message(data, cls)
        p = msg.point if cls is PointStamped else msg.pose.pose.position
        out.setdefault(topic, []).append(
            (msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec, p.x, p.y, p.z))
    return {k: np.array(v) for k, v in out.items()}


def error_against(series, gt):
    inside = (series[:, 0] >= gt[0, 0]) & (series[:, 0] <= gt[-1, 0])
    a = series[inside]
    g = np.column_stack([np.interp(a[:, 0], gt[:, 0], gt[:, k]) for k in (1, 2, 3)])
    return a, np.linalg.norm(a[:, 1:4] - g, axis=1), np.linalg.norm(a[:, 1:3] - g[:, :2], axis=1)


def effective_gain(raw, est):
    """Project the estimate's step onto the innovation, at matching stamps."""
    lookup = {int(s): np.array(p) for s, *p in est[:, :4]}
    seq = [(np.array(p), lookup[int(s)]) for s, *p in raw[:, :4] if int(s) in lookup]
    gains = []
    for i in range(1, len(seq)):
        innovation = seq[i][0] - seq[i - 1][1]
        n = np.linalg.norm(innovation)
        if n > 0.2:                       # skip steps buried in measurement noise
            gains.append(float(np.dot(seq[i][1] - seq[i - 1][1], innovation) / (n * n)))
    return np.median(gains) if gains else float('nan'), len(seq)


def report(uri):
    rec = read(uri)
    for topic in (EST, RAW, GT):
        if topic not in rec:
            sys.exit('%s non contiene %s - registra tutti e tre i topic.' % (uri, topic))

    gt = rec[GT]
    _, keep = np.unique(gt[:, 0], return_index=True)   # stamp duplicati -> uno solo
    gt = gt[np.sort(keep)]
    raw, est = rec[RAW], rec[EST]

    _, raw_err, _ = error_against(raw, gt)
    est_in, est_err, est_h = error_against(est, gt)
    gain, matched = effective_gain(raw, est)
    span = (gt[-1, 0] - gt[0, 0]) / 1e9

    print('%s   (%.0f s di GT)' % (uri, span))
    print('  misure   %5d   mediana %5.2f m   p90 %5.2f m' %
          (len(raw_err), np.median(raw_err), np.percentile(raw_err, 90)))
    print('  stima    %5d   mediana %5.2f m   p90 %5.2f m   max %5.1f m   orizz %5.2f m' %
          (len(est_err), np.median(est_err), np.percentile(est_err, 90),
           est_err.max(), np.median(est_h)))
    print('  uscita a %.1f Hz, %d stamp non monotoni, guadagno efficace %.2f (%d accoppiate)' %
          (len(est_in) / span, (np.diff(est_in[:, 0]) < 0).sum(), gain, matched))

    # Dove sta l'errore: vicino a una misura, o in estrapolazione a ciclo aperto?
    meas = np.sort(raw[:, 0]) / 1e9
    t = est_in[:, 0] / 1e9
    idx = np.searchsorted(meas, t) - 1
    age = np.where(idx >= 0, t - meas[np.clip(idx, 0, len(meas) - 1)], np.nan)
    print('  eta dall ultima misura:')
    for lo, hi in ((0, 0.1), (0.1, 0.5), (0.5, 1.0), (1.0, 3.0), (3.0, np.inf)):
        sel = (age >= lo) & (age < hi)
        if sel.sum() > 5:
            print('    %4.1f-%-5.1f s  %5.1f%% dei campioni   err med %5.2f m'
                  % (lo, hi, 100 * sel.mean(), np.median(est_err[sel])))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for path in sys.argv[1:]:
        report(path)
        print()
