#!/usr/bin/env python3
"""Self-check for the live monitor's GT fallback.

The aligned GT topic only exists while the evaluator runs; the monitor must notice
the silence and plot the target's raw odom instead. Run: python3 test_live_monitor_gt_fallback.py
"""

import rclpy
from nav_msgs.msg import Odometry

from live_monitor import LiveMonitorNode


def _odom(x: float) -> Odometry:
    m = Odometry()
    m.header.stamp.sec = int(x)
    m.pose.pose.position.x = x
    return m


def demo() -> None:
    rclpy.init()
    try:
        node = LiveMonitorNode()
        assert node.target_odom_topic != node.gt_fallback_topic, \
            'config has no aligned GT topic, nothing to fall back from'

        # Before the watchdog fires, raw odom is ignored: the aligned GT may still show up.
        node._gt_fallback_cb(_odom(1.0))
        assert len(node.gt_data) == 0, 'raw odom plotted during the grace period'

        # Watchdog with no aligned GT -> fall back.
        node._gt_watchdog()
        assert node._gt_fallback_active
        node._gt_fallback_cb(_odom(2.0))
        assert len(node.gt_data) == 1, 'fallback did not take over'

        # Aligned GT showing up later wins again.
        node._gt_primary_cb(_odom(3.0))
        assert node._gt_primary_seen and not node._gt_fallback_active
        node._gt_fallback_cb(_odom(4.0))
        assert len(node.gt_data) == 2, 'raw odom still mixed in after the aligned GT came back'

        # Watchdog after the aligned GT arrived must not switch anything.
        node2 = LiveMonitorNode()
        node2._gt_primary_cb(_odom(1.0))
        node2._gt_watchdog()
        assert not node2._gt_fallback_active, 'fell back while the aligned GT was alive'

        node.destroy_node()
        node2.destroy_node()
        print('live monitor GT fallback: OK')
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    demo()
