"""Zc (depth pixel) -> along-ray range conversion in the V2 observation packet.

triangulation_node consumes depth_m as `origin + ray * depth_m`, so the observer
must divide the pixel depth by cos(theta).  Fails if the raw Zc leaks through.
"""
import numpy as np


def normalize(v):
    return v / np.linalg.norm(v)


def along_ray(z_c, sigma, ray_optical):
    cos_theta = float(ray_optical[2])
    if cos_theta <= 1e-6:
        return z_c, sigma
    return z_c / cos_theta, sigma / cos_theta


def demo():
    fx = fy = 380.0
    cx, cy = 320.0, 240.0

    # On-axis detection: range == Zc.
    ray = normalize(np.array([0.0, 0.0, 1.0]))
    r, s = along_ray(5.0, 0.07, ray)
    assert abs(r - 5.0) < 1e-9, r
    assert abs(s - 0.07) < 1e-9, s

    # Edge of frame (u=0): range must exceed Zc and match the geometry exactly.
    ray = normalize(np.array([(0.0 - cx) / fx, (0.0 - cy) / fy, 1.0]))
    r, s = along_ray(5.0, 0.07, ray)
    expected = 5.0 * np.linalg.norm([(0.0 - cx) / fx, (0.0 - cy) / fy, 1.0])
    assert abs(r - expected) < 1e-9, (r, expected)
    assert r > 5.0 * 1.25, f'off-axis correction too small: {r}'
    assert abs(s / r - 0.07 / 5.0) < 1e-12, 'relative sigma must be preserved'

    # Reconstructed 3D point sits at the true Zc plane.
    p = ray * r
    assert abs(p[2] - 5.0) < 1e-9, p

    print('ok')


if __name__ == '__main__':
    demo()
