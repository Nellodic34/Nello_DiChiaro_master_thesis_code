#!/usr/bin/env python3
"""Pre-flight gate for the vision stack. Run on BOTH drones before arming.

Two stages, deliberately separate:

  python3 preflight_check.py           config only, no ROS needed, run it first
  python3 preflight_check.py --live    adds the live attitude / topic checks

The config stage catches the settings that are correct for a bag replay and wrong
for a real flight. The live stage answers the only question you cannot answer
from a file: how far off level each FCU booted this time.
"""
import argparse
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = os.path.join(HERE, 'config', 'runtime_stack.yaml')

R, G, Y, B, N = '\033[31m', '\033[32m', '\033[33m', '\033[1m', '\033[0m'
_fail = []


def ok(m):    print(f'  {G}PASS{N}  {m}')
def warn(m):  print(f'  {Y}WARN{N}  {m}')
def bad(m):   print(f'  {R}FAIL{N}  {m}'); _fail.append(m)


def check_config(c):
    print(f'\n{B}CONFIG{N}  {CFG}')

    # --- gates that are replay-only and actively harmful in flight -----------
    if c.get('bag_profile') != 'none':
        bad(f"bag_profile = {c.get('bag_profile')!r}: injects a bearing_bias fitted to "
            f"ANOTHER power-up. The level reference is set at every boot; 4-8 deg of "
            f"pointing error, 0.31 m per degree at 18 m. Must be 'none'.")
    else:
        ok('bag_profile = none (no stale per-boot bias injected)')

    if c.get('shared_heading'):
        bad('shared_heading = true: rewrites observer1 rays with observer2 yaw. '
            'Only valid with both airframes parked parallel on the ground.')
    else:
        ok('shared_heading = false')

    if c.get('launcher', {}).get('use_sim_time'):
        bad('launcher.use_sim_time = true: the node will wait for a /clock that no real '
            'drone publishes.')
    else:
        ok('launcher.use_sim_time = false')

    vis = c.get('vision', {})
    if vis.get('publish_camera_static_tf'):
        warn('vision.publish_camera_static_tf = true: observer_node will broadcast '
             'uavN/fcu -> uavN/rgbd_link from camera_tf in this YAML. If the real camera '
             'driver also publishes that pair you get two static broadcasters on the same '
             'edge and TF resolves non-deterministically. Set false unless the driver is '
             'silent -- verify with: ros2 run tf2_ros tf2_echo <uav>/fcu <uav>/rgbd_link')

    # --- drone names: two independent places, classic double-edit trap -------
    roles = c.get('uav_roles', {})
    o1, o2, tgt = roles.get('observer1'), roles.get('observer2'), roles.get('target')
    print(f'         roles: observer1={o1}  observer2={o2}  target={tgt}')
    pairs = [('vision.main_image_topic', vis.get('main_image_topic', ''), o1),
             ('vision.observer_image_topic', vis.get('observer_image_topic', ''), o2)]
    for key, topic, uav in pairs:
        if not topic:
            warn(f'{key} empty')
        elif f'/{uav}/' not in topic:
            bad(f'{key} = {topic!r} does not mention {uav!r}. uav_roles and the image '
                f'topics are set in two different places and both must be edited.')
        else:
            ok(f'{key} matches {uav}')

    # --- things that are fine but you want to have SEEN ----------------------
    sr = c.get('observer_self_rejection', {})
    n_drones = c.get('simulation', {}).get('drone_count')
    if n_drones and n_drones >= 3 and not sr.get('enabled'):
        bad(f'{n_drones} drones airborne but observer_self_rejection.enabled = false: '
            f'each observer can lock onto the other observer.')
    else:
        ok(f"observer_self_rejection.enabled = {sr.get('enabled')} (drone_count={n_drones})")

    rate = float(vis.get('max_tf_delta_ms', 0.0))
    if vis.get('allow_tf_fallback'):
        warn(f'vision.allow_tf_fallback = true with max_tf_delta_ms = {rate:g}. A pose '
             f'{rate:g} ms stale during a {50:g} deg/s yaw is {50*rate/1000:.1f} deg of '
             f'bearing error, straight into range as r^2/B.')

    mp = vis.get('model_path', '')
    full = mp if os.path.isabs(mp) else os.path.join(HERE, mp)
    (ok if os.path.exists(full) else bad)(f'vision.model_path {mp} '
                                          f'({"found" if os.path.exists(full) else "MISSING"})')
    print(f"         vision.device = {vis.get('device')}   "
          f"(confirm the detector keeps up: watch the observer node's FPS log)")

    ab = c.get('ekf', {}).get('anisotropic_bearing_noise', {})
    print(f"         anisotropic bearing noise: enabled={ab.get('enabled')} "
          f"az={ab.get('sigma_az_deg')} deg  el={ab.get('sigma_el_deg')} deg")
    if 'normalize_crlb' in c.get('ekf', {}):
        warn('ekf.normalize_crlb is present but is a NO-OP: triangulate_wls never reads it '
             'and always returns the raw CRLB. Ignore the value.')

    df = c.get('depth_fusion', {})
    if not df.get('enabled'):
        print('         depth_fusion.enabled = false (bearing-only, no depth subscriber)')
    else:
        suffix = df.get('depth_topic_suffix', '')
        print(f"         depth_fusion: gating={df.get('gating_enabled')}  "
              f"range_constraint={df.get('fusion_use_range_constraint')}  "
              f"suffix={suffix!r}")
        if any(suffix.endswith(t) for t in ('/compressed', '/compressedDepth', '/theora', '/zstd')):
            bad(f'depth_topic_suffix = {suffix!r}: observer_node subscribes to raw '
                f'sensor_msgs/Image only, there is no image_transport plugin in the node. '
                f'Use the .../image_raw topic.')
        else:
            ok(f'depth_topic_suffix {suffix!r} is a raw Image topic')


def check_live(c, seconds):
    import math
    import numpy as np
    import rclpy
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import CameraInfo, Image

    roles = c.get('uav_roles', {})
    uavs = [roles.get('observer1'), roles.get('observer2'), roles.get('target')]
    uavs = [u for u in uavs if u]

    print(f'\n{B}LIVE{N}  listening {seconds:g} s on /<uav>/estimation_manager/odom_main')
    print('      Park the drones on the SAME flat surface. A common tilt only rotates the')
    print('      solution; it is the DIFFERENCE between the two observers that inflates range.')

    rclpy.init()
    node = Node('preflight_check')
    got = {}
    got_z = {}

    def mk(u):
        def cb(msg):
            q = msg.pose.pose.orientation
            sr_ = 2 * (q.w * q.x + q.y * q.z)
            cr_ = 1 - 2 * (q.x * q.x + q.y * q.y)
            sp_ = max(-1.0, min(1.0, 2 * (q.w * q.y - q.z * q.x)))
            sy_ = 2 * (q.w * q.z + q.x * q.y)
            cy_ = 1 - 2 * (q.y * q.y + q.z * q.z)
            got[u] = (math.degrees(math.atan2(sr_, cr_)),
                      math.degrees(math.asin(sp_)),
                      math.degrees(math.atan2(sy_, cy_)))
            got_z[u] = msg.pose.pose.position.z
        return cb

    for u in uavs:
        node.create_subscription(Odometry, f'/{u}/estimation_manager/odom_main', mk(u), 10)

    # Depth fusion: only subscribe when the master switch is on, mirroring observer_node.
    df = c.get('depth_fusion', {})
    depth_frames, color_info = [], {}
    if df.get('enabled'):
        sensor_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                                history=HistoryPolicy.KEEP_LAST)
        for u in (roles.get('observer1'), roles.get('observer2')):
            if not u:
                continue
            topic = f"/{u}/{df.get('depth_topic_suffix', '')}"
            node.create_subscription(
                Image, topic, lambda m, u=u: depth_frames.append((u, m)), sensor_qos)
            node.create_subscription(
                CameraInfo, f'/{u}/rgbd/color/camera_info',
                lambda m, u=u: color_info.__setitem__(u, (m.width, m.height)), 10)

    end = node.get_clock().now().nanoseconds + int(seconds * 1e9)
    while rclpy.ok() and node.get_clock().now().nanoseconds < end:
        rclpy.spin_once(node, timeout_sec=0.1)

    print(f'\n      {"uav":<10}{"roll":>9}{"pitch":>9}{"yaw":>9}')
    for u in uavs:
        if u not in got:
            bad(f'no odometry from {u} on /{u}/estimation_manager/odom_main')
            continue
        r_, p_, y_ = got[u]
        flag = '  <-- OFF LEVEL' if max(abs(r_), abs(p_)) > 1.0 else ''
        print(f'      {u:<10}{r_:>9.2f}{p_:>9.2f}{y_:>9.2f}{flag}')

    o1, o2 = roles.get('observer1'), roles.get('observer2')
    if o1 in got and o2 in got:
        dr = got[o2][0] - got[o1][0]
        dp = got[o2][1] - got[o1][1]
        dy = (got[o2][2] - got[o1][2] + 180) % 360 - 180
        print(f'\n      differential observer2 - observer1:  '
              f'd_roll {dr:+.2f}  d_pitch {dp:+.2f}  d_yaw {dy:+.2f} deg')
        if max(abs(dr), abs(dp)) > 1.0:
            bad(f'differential tilt {max(abs(dr), abs(dp)):.2f} deg > 1.0. Reboot the worse '
                f'FCU on level ground until it reads level; do NOT compensate in software.')
        else:
            ok('differential tilt within 1.0 deg')
        if abs(dy) > 2.0:
            warn(f'yaw disagreement {dy:+.2f} deg. With dual-antenna RTK heading this should '
                 f'be under ~1.5 deg; if it is not, one heading solution is not fixed yet.')

    # World-origin z datum. Only meaningful with the drones on the SAME flat ground: each
    # reports its height above ITS OWN world_origin, so equal ground + equal z means the
    # origins share a z datum. This is what evaluation.shared_world_origin asserts, and the
    # only way to check it without /tf_static crossing the network.
    shared = c.get('evaluation', {}).get('shared_world_origin', False)
    if len(got_z) >= 2:
        print(f'\n      world-origin z datum (drones must be on the SAME flat ground):')
        for u in uavs:
            if u in got_z:
                print(f'      {u:<10}{got_z[u]:>9.3f} m')
        spread = max(got_z.values()) - min(got_z.values())
        if shared and spread > 0.30:
            bad(f'evaluation.shared_world_origin = true but the odom z values spread by '
                f'{spread:.2f} m. The node is skipping the utm datum hop, so that spread '
                f'goes straight into every remote ray as a constant bias. Either the world '
                f'file z is not actually shared, or the drones are not on the same ground.')
        elif shared:
            ok(f'shared_world_origin = true and z spread {spread:.2f} m (< 0.30): origins agree')
        elif spread <= 0.30:
            print(f'         z spread {spread:.2f} m. Origins look shared; without /tf_static '
                  f'you need evaluation.shared_world_origin = true or observer2 rays are dropped.')
        else:
            print(f'         z spread {spread:.2f} m: origins do NOT share a z datum, the utm '
                  f'hop (and /tf_static) is doing real work. Do not set shared_world_origin.')
    elif shared:
        warn('shared_world_origin = true but fewer than two drones reported odometry, '
             'so the assertion is unverified.')

    if df.get('enabled'):
        print(f"\n      depth fusion on {df.get('depth_topic_suffix')}")
        for u in (roles.get('observer1'), roles.get('observer2')):
            if not u:
                continue
            frames = [m for uu, m in depth_frames if uu == u]
            if not frames:
                bad(f"no depth on /{u}/{df.get('depth_topic_suffix')} while "
                    f"depth_fusion.enabled = true. The node will publish V1 bearing-only "
                    f"packets and the range constraint will never fire.")
                continue
            last = frames[-1]
            hz = len(frames) / seconds
            if last.encoding not in ('16UC1', '32FC1'):
                bad(f'{u} depth encoding {last.encoding!r}: observer_node handles 16UC1 '
                    f'(mm) and 32FC1 (m) only, anything else is read as raw metres.')
            cw_ch = color_info.get(u)
            if cw_ch and cw_ch != (last.width, last.height):
                bad(f'{u} depth {last.width}x{last.height} != color {cw_ch[0]}x{cw_ch[1]}: '
                    f'YOLO bboxes are full-frame COLOR pixels and are indexed straight into '
                    f'the depth array. Use the aligned_depth_to_color stream.')
            elif cw_ch:
                ok(f'{u} depth {last.encoding} {last.width}x{last.height} @ {hz:.1f} Hz, '
                   f'resolution matches color')
            else:
                warn(f'{u} depth {last.encoding} {last.width}x{last.height} @ {hz:.1f} Hz, '
                     f'color camera_info silent so resolution unverified')

            scale = 1e-3 if last.encoding == '16UC1' else 1.0
            dt = np.uint16 if last.encoding == '16UC1' else np.float32
            arr = np.frombuffer(last.data, dtype=dt).astype(np.float32) * scale
            v = arr[np.isfinite(arr) & (arr > 0.05)]
            lo, hi = float(df.get('min_range_m', 0.3)), float(df.get('max_range_m', 10.0))
            if v.size == 0:
                bad(f'{u} depth frame is entirely empty (all holes)')
            else:
                frac = float(((v >= lo) & (v <= hi)).mean())
                print(f'      {u} range p05={np.percentile(v, 5):.2f}  '
                      f'median={np.median(v):.2f}  p95={np.percentile(v, 95):.2f} m  '
                      f'-> {frac:.0%} inside [{lo:g}, {hi:g}]')
                if frac < 0.5:
                    warn(f'{u}: most of the scene is outside [{lo:g}, {hi:g}] m. Depth will '
                         f'be dropped for a target beyond max_range_m; raise it or accept '
                         f'bearing-only at range.')

    node.destroy_node()
    rclpy.shutdown()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--live', action='store_true', help='also check live topics and attitude')
    ap.add_argument('--seconds', type=float, default=5.0)
    a = ap.parse_args()

    c = yaml.safe_load(open(CFG, encoding='utf-8'))
    check_config(c)
    if a.live:
        check_live(c, a.seconds)

    print()
    if _fail:
        print(f'{R}{B}NOT READY TO FLY{N} — {len(_fail)} blocking item(s)')
        return 1
    print(f'{G}{B}config gates clear{N}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
