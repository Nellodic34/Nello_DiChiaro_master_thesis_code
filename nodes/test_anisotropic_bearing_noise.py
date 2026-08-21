#!/usr/bin/env python3
"""Guards the anisotropic bearing-noise term of triangulate_wls().

kf_measurement_noise models the ray uncertainty as a circular cone. It is not:
azimuth error comes from the dual-antenna RTK heading (~1 deg for a 0.55 m
baseline) while elevation error comes from the per-boot level reference
(1.5-3.9 deg measured on the 22-7 bags). ekf.anisotropic_bearing_noise splits
the two. The three properties that must hold:

  1. (a, e, d) is an orthonormal triad, so the per-axis variances ARE the
     eigenvalues of V_vis -- no cross terms sneaking in.
  2. With sigma_az == sigma_el and horizontal rays, the anisotropic path is
     bit-for-bit the isotropic path. Enabling the feature must not silently
     move the estimator when it is configured to be equivalent.
  3. The split actually reaches R_out: raising sigma_el alone must inflate the
     vertical axis of the CRLB and nothing else.

Run: python3 dataset_tools/test_anisotropic_bearing_noise.py
"""
import importlib.util
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location('tn', os.path.join(_HERE, 'triangulation_node.py'))
tn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tn)

BASELINE = 9.3   # uav81 <-> uav8 separation on the v6 bag
RANGE_M = 49.0   # the long shot where the overconfidence showed up


def _horizontal_pair(target_z=0.0):
    """Two observers side by side, target ahead. All rays horizontal => cos(el) = 1."""
    origin1 = np.zeros(3)
    origin2 = np.array([0.0, BASELINE, 0.0])
    target = np.array([RANGE_M, BASELINE / 2.0, target_z])
    return origin1, tn.normalize(target - origin1), origin2, tn.normalize(target - origin2)


def _wls(var_px, aniso=False, var_az=0.0, var_el=0.0, target_z=0.0):
    o1, d1, o2, d2 = _horizontal_pair(target_z)
    return tn.triangulate_wls(
        o1, d1, None, o2, d2, None,
        var_px=var_px,
        use_odom_uncertainty=False,
        distance_dependent_noise=True,
        anisotropic_bearing_noise=aniso,
        bearing_var_az=var_az,
        bearing_var_el=var_el,
        normalize_crlb=False,
    )


def test_triad_is_orthonormal():
    for d in (
        tn.normalize(np.array([1.0, 0.0, 0.0])),
        tn.normalize(np.array([1.0, 2.0, 0.5])),
        tn.normalize(np.array([-3.0, 1.0, -2.0])),
        np.array([0.0, 0.0, 1.0]),          # degenerate: azimuth undefined
    ):
        a, e, cos_el = tn.bearing_frame(d)
        M = np.column_stack([a, e, d])
        assert np.allclose(M.T @ M, np.eye(3), atol=1e-12), f'triad not orthonormal for {d}'
        assert np.isclose(cos_el, np.hypot(d[0], d[1]), atol=1e-12)
    print('  triad orthonormal (incl. vertical ray)   OK')


def test_eigenvalues_are_the_variances():
    """cos^2(el) Jacobian included: a yaw error moves the target by r*cos(el)*d_az."""
    d = tn.normalize(np.array([1.0, 0.0, 1.0]))     # el = 45 deg, cos_el = 1/sqrt(2)
    var_az, var_el, scale = 3e-4, 19e-4, RANGE_M ** 2
    V = tn.bearing_noise_cov(d, var_az, var_el, scale)

    assert np.allclose(V, V.T, atol=1e-15), 'V_vis not symmetric'
    a, e, cos_el = tn.bearing_frame(d)
    assert np.isclose(a @ V @ a, scale * var_az * cos_el ** 2, rtol=1e-12), 'azimuth axis wrong'
    assert np.isclose(e @ V @ e, scale * var_el, rtol=1e-12), 'elevation axis wrong'
    assert np.isclose(a @ V @ e, 0.0, atol=1e-12), 'az/el axes are coupled'
    print('  eigen-axes carry exactly var_az*cos^2(el) / var_el   OK')


def test_isotropic_equivalence():
    """sigma_az == sigma_el on horizontal rays must reproduce the isotropic path."""
    var = 0.0019
    p_iso, R_iso = _wls(var)
    p_ani, R_ani = _wls(var, aniso=True, var_az=var, var_el=var)
    assert np.allclose(p_iso, p_ani, atol=1e-9), 'position moved when it should not have'
    assert np.allclose(R_iso, R_ani, rtol=1e-9), 'CRLB moved when it should not have'
    print('  equal sigmas == isotropic path            OK')


def test_split_reaches_R_out():
    """Rays are horizontal here, so the elevation axis of every ray is vertical."""
    tall = 3.0 * np.deg2rad(1.0) ** 2        # sigma_el = 3 deg, sigma_az = 1 deg
    small = np.deg2rad(1.0) ** 2
    _, R_flat = _wls(0.0, aniso=True, var_az=small, var_el=small)
    _, R_tall = _wls(0.0, aniso=True, var_az=small, var_el=tall)

    assert R_tall[2, 2] > 2.5 * R_flat[2, 2], 'sigma_el did not inflate the vertical axis'
    assert np.isclose(R_tall[0, 0], R_flat[0, 0], rtol=1e-6), 'sigma_el leaked into the depth axis'
    assert np.isclose(R_tall[1, 1], R_flat[1, 1], rtol=1e-6), 'sigma_el leaked into the lateral axis'
    print('  sigma_el inflates only the vertical axis  OK')


def test_attitude_errors_land_on_the_right_axis():
    """The claim that justifies the whole feature: yaw and tilt do NOT mix.

    sigma_az is fed by the RTK heading and sigma_el by the boot level reference,
    which is only meaningful if a yaw error stays out of the elevation axis and a
    pitch error stays out of the azimuth axis. To first order -- which is the
    order the covariance model works at -- a small attitude error ``w`` moves the
    bearing by ``dd = w x d``, and that lands as:

        yaw   (w = z_world) -> cos(el) on a,   EXACTLY 0 on e
        pitch (w = y_body)  -> EXACTLY 0 on a, 1 on e
        tilt along the LOS  -> sin(el) on a,   0 on e   <- the only cross-term

    So the two budgets are separable, exactly, for a target at the observer's
    altitude, and degrade only as sin(el) when it is not.
    """
    for el_deg in (0.0, 5.0, 20.0):
        el = np.deg2rad(el_deg)
        d = np.array([np.cos(el), 0.0, np.sin(el)])
        a, e, cos_el = tn.bearing_frame(d)

        yaw = np.cross(np.array([0.0, 0.0, 1.0]), d)
        assert np.isclose(yaw @ a, cos_el, atol=1e-12), 'yaw lost its cos(el) scaling'
        assert np.isclose(yaw @ e, 0.0, atol=1e-12), 'yaw leaked into elevation'

        pitch = np.cross(np.array([0.0, 1.0, 0.0]), d)
        assert np.isclose(abs(pitch @ e), 1.0, atol=1e-12), 'body pitch lost the elevation axis'
        assert np.isclose(pitch @ a, 0.0, atol=1e-12), 'body pitch leaked into azimuth'

        tilt_los = np.cross(np.array([1.0, 0.0, 0.0]), d)   # horizontal, along the LOS
        assert np.isclose(abs(tilt_los @ a), np.sin(el), atol=1e-12), 'sin(el) azimuth leak wrong'
        assert np.isclose(tilt_los @ e, 0.0, atol=1e-12), 'LOS tilt should not touch elevation'
    print('  yaw -> azimuth only, pitch -> elevation only  OK')


def test_radial_filler_never_reaches_R_out():
    """var_r exists only to keep V_i invertible; Pi_i must cancel it exactly."""
    o1, d1, o2, d2 = _horizontal_pair()
    var_az, var_el = np.deg2rad(1.32) ** 2, np.deg2rad(2.63) ** 2
    real = tn.bearing_noise_cov

    def with_filler(var_r):
        def patched(dirv, va, ve, scale):
            a, e, cos_el = tn.bearing_frame(dirv)
            return scale * (va * cos_el ** 2 * np.outer(a, a)
                            + ve * np.outer(e, e)
                            + var_r * np.outer(dirv, dirv))
        tn.bearing_noise_cov = patched
        try:
            return tn.triangulate_wls(
                o1, d1, None, o2, d2, None, var_px=0.0, use_odom_uncertainty=False,
                anisotropic_bearing_noise=True,
                bearing_var_az=var_az, bearing_var_el=var_el)[1]
        finally:
            tn.bearing_noise_cov = real

    R_ref = with_filler(0.5 * (var_az + var_el))
    for var_r in (1e-9, 1e3):
        R = with_filler(var_r)
        rel = np.abs(R - R_ref).max() / np.abs(R_ref).max()
        assert rel < 1e-9, f'radial filler {var_r} moved R_out by {rel:.2e}'
    print('  radial filler cancelled by Pi_i (1e-9 .. 1e3)  OK')


def main():
    test_triad_is_orthonormal()
    test_eigenvalues_are_the_variances()
    test_isotropic_equivalence()
    test_split_reaches_R_out()
    test_attitude_errors_land_on_the_right_axis()
    test_radial_filler_never_reaches_R_out()
    print('OK')


if __name__ == '__main__':
    main()
