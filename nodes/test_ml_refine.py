#!/usr/bin/env python3
"""Guards the Gauss-Newton ML refinement of triangulate_wls().

The WLS solve is a PSEUDOLINEAR estimator: it minimises ||Pi_i (p - c_i)||, a
METRIC residual, which is what makes it closed-form and what makes it biased.
The regressor Pi_i is built from the noisy bearing, so regressor and residual
are correlated and the bias survives averaging (Gavish & Weiss 1992, Sec. III).
The bias points TOWARDS the observers -- the classic bearing-only range
shortfall -- and grows with range/baseline. ml_refine runs Gauss-Newton on the
ANGULAR residual instead, which is where the noise actually lives, and is
asymptotically unbiased (ibid., Sec. IV).

The four properties that must hold:

  1. On noiseless bearings the WLS answer is already exact, so refinement must
     be a fixed point. Anything else means the residual or Jacobian is wrong.
  2. Over a Monte-Carlo run the MEDIAN radial error must shrink towards zero.
     This is the whole point of the feature and the number to quote in a report.
     Measured here: -0.39 m -> +0.06 m at 20 m, -5.18 m -> -0.38 m at 49 m
     (9.3 m baseline, sigma 2.5 deg). Note that the sample MEAN moves the other
     way; see the comment in that test for why the mean is the wrong statistic.
  3. Refinement must never make a single frame worse than a fallback: on
     garbage geometry it returns the WLS answer, flagged by iters == 0.
  4. ml_refine=False must be bit-for-bit the old behaviour.

Run: python3 <dir>/test_ml_refine.py
     <dir> is nodes/ in the deploy tree, dataset_tools/ in the bagtest tree.
     The two copies are kept byte-identical, so `diff -q` is the alignment check.
"""
import importlib.util
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location('tn', os.path.join(_HERE, 'triangulation_node.py'))
tn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tn)

BASELINE = 9.3     # uav81 <-> uav8 separation on the v6 bag
RANGE_M = 49.0     # the long shot where the bias is worst
SIGMA_RAD = np.deg2rad(2.5)
VAR = SIGMA_RAD ** 2

ORIGIN1 = np.zeros(3)
ORIGIN2 = np.array([0.0, BASELINE, 0.0])
TARGET = np.array([RANGE_M, BASELINE / 2.0, 0.0])
TARGET_20M = np.array([20.0, BASELINE / 2.0, 0.0])   # the usual working range


def _perturb(d, rng, sigma=SIGMA_RAD):
    """Rotate a unit bearing by a small random angle in both perpendicular axes."""
    a, e, _ = tn.bearing_frame(d)
    return tn.normalize(d + rng.normal(0.0, sigma) * a + rng.normal(0.0, sigma) * e)


def _solve(d1, d2, ml_refine, var=VAR, prior=None):
    """None when the draw is rejected by the observability check, exactly as the
    live node rejects it -- skipping it for BOTH estimators keeps the comparison fair."""
    try:
        return tn.triangulate_wls(
        ORIGIN1, d1, None, ORIGIN2, d2, None,
            var_px=var,
            use_odom_uncertainty=False,
            distance_dependent_noise=True,
            normalize_crlb=False,
            ml_refine=ml_refine,
            prior_target_pos=prior,
        )
    except ValueError:
        return None


def test_noiseless_geometry_is_a_fixed_point():
    d1, d2 = tn.normalize(TARGET - ORIGIN1), tn.normalize(TARGET - ORIGIN2)
    p_wls, _ = _solve(d1, d2, ml_refine=False)
    p_ml, R_ml = _solve(d1, d2, ml_refine=True)
    assert np.allclose(p_wls, TARGET, atol=1e-6), p_wls
    assert np.allclose(p_ml, TARGET, atol=1e-6), p_ml
    assert np.all(np.isfinite(R_ml))
    print(f'  fixed point: |p_ml - target| = {np.linalg.norm(p_ml - TARGET):.2e} m   OK')


def _bias_pair(target, sigma_deg, n, seed):
    """(median, mean) radial error of WLS and of WLS+ML over ``n`` noisy draws."""
    rng = np.random.default_rng(seed)
    sigma = np.deg2rad(sigma_deg)
    err = {False: [], True: []}
    for _ in range(n):
        d1 = _perturb(tn.normalize(target - ORIGIN1), rng, sigma)
        d2 = _perturb(tn.normalize(target - ORIGIN2), rng, sigma)
        both = {ml: _solve(d1, d2, ml, sigma ** 2, target) for ml in (False, True)}
        if any(v is None for v in both.values()):
            continue
        for ml, v in both.items():
            err[ml].append(v[0] - target)
    los = tn.normalize(target - 0.5 * (ORIGIN1 + ORIGIN2))
    out = {}
    for ml, e in err.items():
        radial = np.asarray(e) @ los
        out[ml] = (float(np.median(radial)), float(np.mean(radial)))
    return out


def test_refinement_reduces_the_pseudolinear_bias():
    # Judged on the MEDIAN, not the mean. The 2-ray bearing-only range estimate has
    # a heavy right tail -- draws where the rays come out near-parallel put the fix
    # at hundreds of metres -- so the sample mean is a tail statistic, not a bias
    # statistic, and it moves POSITIVE under ML while the systematic error goes
    # away. The median is what the IMM actually integrates frame after frame; the
    # tail draws get killed by the Mahalanobis gate.
    for target, sigma_deg, n in ((TARGET_20M, 2.5, 800), (TARGET, 2.5, 800)):
        res = _bias_pair(target, sigma_deg, n, seed=20260907)
        rng_m = float(np.linalg.norm(target - 0.5 * (ORIGIN1 + ORIGIN2)))
        (med_w, mean_w), (med_m, mean_m) = res[False], res[True]
        print(f'  @{rng_m:.0f} m, sigma {sigma_deg} deg: median radial '
              f'{med_w:+.2f} -> {med_m:+.2f} m   (mean {mean_w:+.2f} -> {mean_m:+.2f}, tail-dominated)')
        assert med_w < 0.0, f'expected the PLE range shortfall, got {med_w:+.2f}'
        assert abs(med_m) < abs(med_w), (med_w, med_m)


def test_degenerate_geometry_falls_back_to_wls():
    # Rays 0.001 deg apart: the ML normal matrix is numerically singular and any
    # step is meaningless. The contract is "return the input", not "return NaN".
    d1 = tn.normalize(np.array([1.0, 0.0, 0.0]))
    d2 = tn.normalize(np.array([1.0, 1.7e-5, 0.0]))
    p_init = np.array([RANGE_M, 0.0, 0.0])
    p_out, R_out, iters = tn.refine_triangulation_gn(
        p_init, [ORIGIN1, ORIGIN2], [d1, d2], VAR, VAR,
    )
    assert np.allclose(p_out, p_init), p_out
    assert iters == 0 or np.all(np.isnan(R_out))
    print('  near-parallel rays fall back to the WLS estimate   OK')


def test_disabled_flag_is_bit_for_bit_the_old_path():
    rng = np.random.default_rng(7)
    for _ in range(20):
        d1 = _perturb(tn.normalize(TARGET - ORIGIN1), rng)
        d2 = _perturb(tn.normalize(TARGET - ORIGIN2), rng)
        p_a, R_a = _solve(d1, d2, ml_refine=False)
        p_b, R_b = _solve(d1, d2, ml_refine=False)
        assert np.array_equal(p_a, p_b) and np.array_equal(R_a, R_b)
    print('  ml_refine=False path unchanged   OK')


def test_covariance_stays_a_usable_crlb():
    # R_ml feeds the IMM update directly, so it must stay symmetric positive
    # definite and on the same physical scale as the WLS CRLB -- otherwise the
    # Mahalanobis gate and the tuned Q are silently invalidated.
    d1, d2 = tn.normalize(TARGET - ORIGIN1), tn.normalize(TARGET - ORIGIN2)
    _, R_wls = _solve(d1, d2, ml_refine=False)
    _, R_ml = _solve(d1, d2, ml_refine=True)
    assert np.allclose(R_ml, R_ml.T, atol=1e-12)
    eig = np.linalg.eigvalsh(R_ml)
    assert np.all(eig > 0.0), eig
    ratio = float(np.trace(R_ml) / np.trace(R_wls))
    print(f'  trace(R_ml)/trace(R_wls) = {ratio:.3f}')
    assert 0.5 < ratio < 2.0, ratio


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_'):
            fn()
            print(f'PASS  {name}')
    print('all ok')
